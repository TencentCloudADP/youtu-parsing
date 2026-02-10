import os
import json
import torch
import argparse
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

try:
    from .preprocessing.angle_corrector import AngleCorrector
except ImportError:
    from preprocessing.angle_corrector import AngleCorrector

from youtu_parsing_utils import (
    MIN_PIXELS, MAX_PIXELS, IMAGE_EXT, PDF_EXT,
    PROMPT_DICT,
    parse_layout_str,
    convert_table_ostl_to_html,
    load_images_from_pdf, load_image,
    parse_figtext_str,
    build_hierarchy_json,
    draw_layout_on_image,
    box_to_str, clean_repeated_substrings,
    parse_batch_results, categorize_layout_items,
    scale_bounding_boxes
)


class YoutuOCRParserHF:
    
    def __init__(self, 
                 model_path: str,
                 enable_angle_correct: bool = False,
                 angle_correct_model_path: str = '',
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS):
        """
        Initialize the OCR parser with model and configuration.
        
        Args:
            model_path: Path to the Youtu-VL model
            enable_angle_correct: Whether to enable angle correction
            angle_correct_model_path: Path to angle correction model
            min_pixels: Minimum pixel constraint for images
            max_pixels: Maximum pixel constraint for images
        """
        # Load the main HuggingFace model and processor
        self._load_hf_model(model_path)
        
        # Setup angle correction module if enabled
        self._setup_angle_correction(enable_angle_correct, angle_correct_model_path)
        
        # Validate pixel constraints are within acceptable ranges
        self._validate_pixel_constraints(min_pixels, max_pixels)
        
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _load_hf_model(self, model_path: str) -> None:
        """Load HuggingFace model and processor."""
        # Load model with flash attention and bfloat16 for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add special tokens for masking and separation
        self.processor.tokenizer.add_tokens(["<mask>", "<sep>"])
        self.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<mask>")

    def _setup_angle_correction(self, enable_angle_correct: bool, model_path: str) -> None:
        """Setup angle correction module if enabled."""
        self.enable_angle_correct = enable_angle_correct
        if self.enable_angle_correct:
            self.angle_corrector = AngleCorrector(model_path)

    def _validate_pixel_constraints(self, min_pixels: int, max_pixels: int) -> None:
        """Validate pixel constraints are within acceptable ranges."""
        assert min_pixels is None or min_pixels >= MIN_PIXELS
        assert max_pixels is None or max_pixels <= MAX_PIXELS

    def _get_prompt(self, prompt_mode: str, bbox: Optional[List[float]] = None) -> str:
        """Get prompt template and format with bbox if needed."""
        prompt = PROMPT_DICT[prompt_mode]
        
        # These modes require region coordinates to be injected into prompt
        region_specific_modes = ["text_recognize", "title_recognize", "table_recognize", 
                                 "chart_logic_recognize", "chart_data_recognize"]
        if prompt_mode in region_specific_modes:
            assert bbox is not None, f"Bounding box required for {prompt_mode}"
            # Format prompt with bbox coordinates [x1, y1, x2, y2]
            prompt = prompt.format(bbox[0], bbox[1], bbox[2], bbox[3])
        
        return prompt

    def _compute_image_scaling(self, image: Image.Image, inputs: Dict) -> Tuple[float, float]:
        """Compute scaling factors between original and processed images."""
        raw_image_w, raw_image_h = image.size
        # Spatial shapes represent patch dimensions; multiply by 16 to get pixel dimensions
        inp_w = inputs["spatial_shapes"][0][1].item() * 16
        inp_h = inputs["spatial_shapes"][0][0].item() * 16
        scale_w = raw_image_w / inp_w
        scale_h = raw_image_h / inp_h
        return scale_w, scale_h

    def _inference_with_hf(self, 
                          image: Image.Image, 
                          prompt: str, 
                          image_embeds: Optional[torch.Tensor] = None, 
                          max_new_tokens: int = 8192) -> Tuple[str, float, float, torch.Tensor]:
        """
        Perform inference with the Hugging Face model.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt for the model
            image_embeds: Pre-computed image embeddings (optional)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (response_text, scale_w, scale_h, image_embeds)
        """
        with torch.no_grad():
            # Prepare chat messages in OpenAI format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template to format messages
            prompt_str = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs (tokenize and encode image)
            inputs = self.processor(
                text=prompt_str,
                images=[image],
                padding=True,
                return_tensors="pt",
                max_image_patches=self.max_pixels
            ).to(self.model.device)

            # Compute scaling factors for bbox coordinate mapping
            scale_w, scale_h = self._compute_image_scaling(image, inputs)

            # Get or compute image embeddings (reuse if provided)
            if image_embeds is None:
                image_embeds = self.model.get_visiual_features(
                    inputs.pixel_values, 
                    inputs.pixel_attention_mask, 
                    inputs.spatial_shapes
                )

            # Generate response using parallel decoder
            generated_ids = self.model.generate_parallel_decoder(
                inputs, 
                image_embeds, 
                self.mask_token_id, 
                max_new_tokens=max_new_tokens, 
                verbose=False
            )
            
            response = self.processor.decode(generated_ids).rstrip('\n')

            return response, scale_w, scale_h, image_embeds

    def _infer_bbox_query_batch(self, 
                               image: Image.Image, 
                               bbox_list: List[List[float]], 
                               layout_types: List[str], 
                               image_embeds: Optional[torch.Tensor] = None, 
                               max_batch_size: int = 5) -> Tuple[List[str], torch.Tensor]:
        """Batch inference for multiple bounding boxes."""
        if not bbox_list:
            return [], image_embeds
        
        # Limit batch size to avoid context overflow
        bbox_list = bbox_list[:max_batch_size]
        layout_types = layout_types[:max_batch_size]
        
        # Single bbox: use simple recognition query
        if len(bbox_list) == 1:
            prompt_box = self._get_prompt("text_recognize", bbox_list[0])
            result, _, _, new_embeds = self._inference_with_hf(image, prompt_box, image_embeds)
            return [result], new_embeds if image_embeds is None else image_embeds
        
        # Multiple bboxes: combine into batch query for efficiency
        box_query_parts = [box_to_str(bbox, layout_type) 
                          for bbox, layout_type in zip(bbox_list, layout_types)]
        
        combined_query = (
            "Based on the given input field coordinates and layout type, "
            "identify and extract the content within the specified region. "
            "Formulas shall be represented in LaTeX notation, and tables "
            "shall be structured in OTSL format: " + "|".join(box_query_parts)
        )
        
        output_text, _, _, new_embeds = self._inference_with_hf(image, combined_query, image_embeds)
        
        # Parse model output into individual bbox results
        return parse_batch_results(output_text, len(bbox_list)), \
               new_embeds if image_embeds is None else image_embeds

    def _process_figure_items(self, 
                             image: Image.Image, 
                             layout_items: List[List], 
                             figure_indices: List[int], 
                             recognize_results: List[str]) -> None:
        """Process figure items by cropping and recognizing text."""
        for i in figure_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                # Crop the figure region from the full image
                crop_image = image.crop([int(x1), int(y1), int(x2), int(y2)])
                
                # Recognize figure content
                prompt_box = self._get_prompt("figure_recognize")
                response_box_content, _, _, _ = self._inference_with_hf(crop_image, prompt_box)
                # Remove repeated patterns that may appear in model output
                response_box_content = clean_repeated_substrings(
                    response_box_content) if response_box_content else ""

                # Parse and combine figure text
                figtext_items = parse_figtext_str(response_box_content)
                recognize_results[i] = ''.join(item['content'] + '\n' for item in figtext_items)
            except Exception as e:
                print(f"FIGURE crop recognition error: {e}")
                recognize_results[i] = ""
    
    def _process_chart_items(self, 
                            image: Image.Image, 
                            layout_items: List[List], 
                            chart_indices: List[int], 
                            recognize_results: List[str], 
                            image_embeds: torch.Tensor) -> None:
        """Process chart items with region-based recognition."""
        for i in chart_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                prompt_box = self._get_prompt(f"{layout_type.lower()}_recognize", [x1, y1, x2, y2])
                response_box_content, _, _, _ = self._inference_with_hf(image, prompt_box, image_embeds)
                recognize_results[i] = response_box_content if response_box_content else ""
            except Exception as e:
                print(f"CHART recognition error: {e}")
                recognize_results[i] = ""
    
    def _process_seal_items(self, 
                            image: Image.Image, 
                            layout_items: List[List], 
                            seal_indices: List[int], 
                            recognize_results: List[str], 
                            image_embeds: torch.Tensor) -> None:
        """Process seal items with region-based recognition."""
        for i in seal_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                prompt_box = self._get_prompt("seal_recognize", [x1, y1, x2, y2])
                response_box_content, _, _, _ = self._inference_with_hf(image, prompt_box, image_embeds)
                recognize_results[i] = response_box_content if response_box_content else ""
            except Exception as e:
                print(f"SEAL recognition error: {e}")
                recognize_results[i] = ""

    def _process_table_items(self, 
                            image: Image.Image, 
                            layout_items: List[List], 
                            table_indices: List[int], 
                            recognize_results: List[str], 
                            image_embeds: torch.Tensor) -> None:
        """Process table items and convert to HTML format."""
        for i in table_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                prompt_box = self._get_prompt("table_recognize", [x1, y1, x2, y2])
                response_box_content, _, _, _ = self._inference_with_hf(image, prompt_box, image_embeds)
                # Convert OTSL (Open Table Structure Language) format to HTML
                recognize_results[i] = convert_table_ostl_to_html(response_box_content) if response_box_content else ""
            except Exception as e:
                print(f"TABLE recognition or HTML conversion error: {e}")
                recognize_results[i] = ""

    def _process_ocr_items_batch(self, 
                                image: Image.Image, 
                                layout_items: List[List], 
                                ocr_indices: List[int], 
                                recognize_results: List[str], 
                                image_embeds: torch.Tensor, 
                                batch_size: int = 5) -> None:
        """Process OCR items in batches for efficiency."""
        # Split indices into batches
        for batch_start in range(0, len(ocr_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(ocr_indices))
            batch_indices = ocr_indices[batch_start:batch_end]
            
            # Extract bbox and type for current batch
            batch_bboxes = [[*layout_items[idx][:4]] for idx in batch_indices]
            batch_types = [layout_items[idx][4] for idx in batch_indices]
            
            try:
                # Batch inference for all items in current batch
                batch_results, _ = self._infer_bbox_query_batch(
                    image, batch_bboxes, batch_types, 
                    image_embeds=image_embeds, 
                    max_batch_size=batch_size
                )
                
                # Validate result count matches input count
                if len(batch_results) != len(batch_bboxes):
                    print(f"Warning: Batch result count ({len(batch_results)}) "
                          f"doesn't match input count ({len(batch_bboxes)})")
                    batch_results.extend([""] * (len(batch_bboxes) - len(batch_results)))
                
                # Map results back to original indices
                for j, idx in enumerate(batch_indices):
                    recognize_results[idx] = batch_results[j] if j < len(batch_results) else ""
            except Exception as e:
                print(f"Batch OCR recognition error: {e}")
                for idx in batch_indices:
                    recognize_results[idx] = ""
    
    def _process_hierarchical_layout(self, 
                                   image: Image.Image, 
                                   layout_items: List[List], 
                                   image_embeds: torch.Tensor) -> Tuple[str, Dict[str, int]]:
        """
        Process hierarchical layout relationships among layout items.
        
        Converts bbox_2d to string format for inference in the following format:
        <x_211><y_189><x_362><y_245><LAYOUT_HEADER><000>\n<x_675><y_173><x_1041><y_267><LAYOUT_HEADER><001>
        
        Args:
            image: Input PIL Image
            layout_items: List of layout items, each containing [x1, y1, x2, y2, layout_type]
            image_embeds: Pre-computed image embeddings
            
        Returns:
            Tuple of (response_text, new_to_orig_idx):
            - response_text: Hierarchical relationship string from model
            - new_to_orig_idx: Mapping from new index (str) to original index (int)
        """
        if not layout_items:
            return '', {}
        
        try:
            # Build hierarchical relationship prompt
            bbox_string_list = []
            # Mapping from new index to original index: new_index -> original_idx
            original_index_map = {}
            
            for current_idx, layout_element in enumerate(layout_items):
                x1, y1, x2, y2, element_type = layout_element
                
                # Standardize label format
                if not element_type.startswith("<"):
                    element_type = f"<LAYOUT_{element_type.upper()}>"
                
                bbox_format_string = f'<x_{x1}><y_{y1}><x_{x2}><y_{y2}{element_type}<{current_idx:03d}>'
                bbox_string_list.append(bbox_format_string)
                
                # Record mapping: new index -> original index
                original_index_map[f"{current_idx:03d}"] = current_idx
            
            if not bbox_string_list:
                return '', {}
            
            # Combine all bbox strings
            full_layout_prompt = "\n".join(bbox_string_list)
            
            # Create hierarchical analysis query
            complete_query = full_layout_prompt + PROMPT_DICT['hierarchy_recognize']
            
            # Perform hierarchical relationship inference
            inference_result, _, _, _ = self._inference_with_hf(image, complete_query, image_embeds)
            
            return inference_result, original_index_map
            
        except Exception as e:
            print(f"HIERARCHICAL layout processing error: {e}")
            return '', {}

    def _parse_document(self, 
                            image: Image.Image, 
                            batch_size: int = 5) -> Tuple[List[Dict], float, float, torch.Tensor, Dict[str, Any]]:
        """
        Parse document with layout detection and content extraction.
        
        Returns:
            Tuple of (result, scale_w, scale_h, image_embeds, hierarchy_json)
        """
        # Step 1: Detect layout regions and their types
        prompt_layout = self._get_prompt("layout_detect")
        response_layout, scale_w, scale_h, image_embeds = self._inference_with_hf(image, prompt_layout)
        print(f"Layout detection response: {response_layout}")
        
        layout_items = parse_layout_str(response_layout)
        if not layout_items:
            return [], scale_w, scale_h, image_embeds, '', {}, [], []
        
        # Step 2: Categorize items by processing type (figure/chart/table/ocr)
        categories = categorize_layout_items(layout_items)
        
        # Initialize results storage
        recognize_results = [""] * len(layout_items)
        
        # Step 3: Process each category with appropriate method
        self._process_figure_items(
            image, layout_items, categories['figure_indices'], recognize_results
        )
        
        # Process chart elements 
        self._process_chart_items(
            image, layout_items, categories['chart_indices'],
            recognize_results, image_embeds
        )
        
        # Process table elements with OTSL format conversion to HTML
        self._process_table_items(
            image, layout_items, categories['table_indices'], 
            recognize_results, image_embeds
        )

        # Process seal elements 
        self._process_seal_items(
            image, layout_items, categories['seal_indices'], 
            recognize_results, image_embeds
        )
        
        # Process text elements using batch OCR for efficiency
        self._process_ocr_items_batch(
            image, layout_items, categories['ocr_indices'], 
            recognize_results, image_embeds, batch_size
        )

        hierarchy_results, new_to_orig_idx = self._process_hierarchical_layout(image, layout_items, image_embeds)
        
        # Step 4: Build final structured results
        result = []
        for i, layout_item in enumerate(layout_items):
            x1, y1, x2, y2, layout_type = layout_item
            item = {
                "type": layout_type,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }
            if recognize_results[i]:
                item["content"] = recognize_results[i]
            result.append(item)
        
        return (
            result, 
            scale_w, 
            scale_h, 
            image_embeds, 
            hierarchy_results, 
            new_to_orig_idx, 
            layout_items, 
            recognize_results
        )

    def _parse_single_image(self, 
                           image_input: Image.Image, 
                           enable_angle_corrector: bool = True, 
                           batch_size: int = 5) -> Tuple[List[Dict], Optional[float], Dict[str, Any]]:
        """
        Parse single image and extract layout and content.
        
        Returns:
            Tuple of (result, page_angle, hierarchy_json)
        """
        # Preprocessing: apply angle correction if enabled
        page_angle = None
        if enable_angle_corrector and self.enable_angle_correct:
            image, M, page_angle = self.angle_corrector(image_input)
        else:
            image = image_input
            M = None

        # Parse document with layout detection and content recognition
        (
            result, 
            scale_w, 
            scale_h, 
            image_embeds, 
            hierarchy_results, 
            new_to_orig_idx, 
            layout_items, 
            recognize_results
        ) = self._parse_document(image, batch_size)
        
        # Scale bounding boxes back to original image dimensions
        scale_bounding_boxes(result, scale_w, scale_h, image.size)
        
        # Reverse angle correction transformation on coordinates
        if enable_angle_corrector and self.enable_angle_correct and M is not None:
            for i in range(len(result)):
                result[i]['bbox'] = self.angle_corrector.coord_inverse_rotation(result[i]['bbox'], M)
        

        # Build hierarchical JSON structure from layout analysis results
        try:
            hierarchy_json = build_hierarchy_json(
                hierarchy_results, result, new_to_orig_idx
            )
        except Exception as e:
            print(f"[WARNING] Failed to build hierarchy JSON: {e}")
            hierarchy_json = {
                "total_nodes": 0,
                "root_count": 0,
                "max_level": 0,
                "nodes": []
            }

        return result, page_angle, hierarchy_json

    def parse_pdf(self, 
                  input_path: str, 
                  dpi: int = 200, 
                  start_page_idx: int = 0, 
                  end_page_idx: int = -1) -> List[Dict]:
        """Parse PDF file by converting to images and processing each page."""
        filename, ext = os.path.splitext(os.path.basename(input_path))
        
        # Convert PDF pages to images with specified DPI
        images = load_images_from_pdf(
            pdf_path=input_path,
            dpi=dpi,
            start_page_idx=start_page_idx,
            end_page_idx=end_page_idx
        )

        # Process each page sequentially
        result_pdf = []
        for i, image in enumerate(images):
            page_result, page_angle, hierarchy_json = self._parse_single_image(image)
            result_page = {
                "filename": filename,
                "page_idx": i + start_page_idx,
                "image": image,
                "page_result": page_result,
                "page_angle": page_angle,
                "hierarchy": hierarchy_json,
            }
            result_pdf.append(result_page)
        
        return result_pdf

    def parse_image(self, input_path: str) -> List[Dict]:
        """Parse single image file."""
        filename, ext = os.path.splitext(os.path.basename(input_path))
        image = load_image(input_path)

        # Parse the image
        page_result, page_angle, hierarchy_json = self._parse_single_image(
            image, enable_angle_corrector=self.enable_angle_correct
        )

        return [{
            "filename": filename,
            "image": image,
            "page_result": page_result,
            "page_angle": page_angle,
            "hierarchy": hierarchy_json,
        }]

    def _generate_output_path(self, input_path: str, output_dir: str) -> str:
        """Generate output file path."""
        json_path = os.path.splitext(os.path.basename(input_path))[0] + '.json'
        json_path = os.path.join(output_dir, json_path)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return json_path

    def parse_file(self, 
                   input_path: str, 
                   output_dir: str, 
                   dpi: int = 200, 
                   start_page_idx: int = 0, 
                   end_page_idx: int = -1) -> None:
        """Parse file and save results in multiple formats (JSON, MD, PNG)."""
        filename, ext = os.path.splitext(os.path.basename(input_path))

        # Dispatch to appropriate parser based on file extension
        if ext in PDF_EXT:
            results = self.parse_pdf(
                input_path, dpi=dpi, 
                start_page_idx=start_page_idx, end_page_idx=end_page_idx
            )
        elif ext in IMAGE_EXT:
            results = self.parse_image(input_path)
        else:
            raise ValueError(f"Input file extension {ext} not supported.")

        # Save each page result in multiple formats: JSON, Hierarchy JSON, Markdown, and PNG visualization
        os.makedirs(output_dir, exist_ok=True)
        for result in results:
            json_path = self._generate_output_path(input_path, output_dir)
            
            # Save as JSON
            with open(json_path, "w", encoding='utf-8') as jf:
                json.dump(result["page_result"], jf, ensure_ascii=False, indent=2)
            
            # Save hierarchy as separate JSON file
            if "hierarchy" in result and result["hierarchy"]:
                hierarchy_path = os.path.splitext(json_path)[0] + "_hierarchy.json"
                with open(hierarchy_path, "w", encoding='utf-8') as hf:
                    json.dump(result["hierarchy"], hf, ensure_ascii=False, indent=2)
            
            # Save as Markdown (text content only)
            with open(os.path.splitext(json_path)[0] + ".md", "w", encoding='utf-8') as mf:
                mf.write("\n\n".join([r["content"] for r in result["page_result"] if "content" in r]))
            
            # Save layout visualization with bounding boxes
            img = draw_layout_on_image(result['image'], result['page_result'])
            img.save(os.path.splitext(json_path)[0] + "_layout.png")

    def parse_list(self, list_path: str, output_dir: str) -> None:
        """Parse multiple files listed in a text file."""
        # Read file paths from the list file (one path per line)
        with open(list_path, 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Process each file with progress tracking
        for file_path in tqdm(file_list, desc="Processing files"):
            try:
                self.parse_file(file_path, output_dir)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Youtu OCR Parser")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_weights/Youtu-Parsing",
        help="Path to the Youtu-VL model directory"
    )

    parser.add_argument(
        "--enable_angle_correct",
        action="store_true",
        default=False,
        help="Enable angle correction preprocessing"
    )

    parser.add_argument(
        "--angle_correct_model_path",
        type=str,
        default="./model_weights/angle_preprocess/",
        help="Path to angle correction model"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input image or PDF file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion"
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize OCR parser with model and configuration
    ocr_parser = YoutuOCRParserHF(
        model_path=args.model_path, 
        enable_angle_correct=args.enable_angle_correct,
        angle_correct_model_path=args.angle_correct_model_path
    )
    
    # Parse the specified file and save results to output directory
    ocr_parser.parse_file(
        input_path=args.input_path,
        output_dir=args.output_dir,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
