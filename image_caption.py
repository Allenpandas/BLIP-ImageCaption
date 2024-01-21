import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def load_blip_model(model_url, image_size, device):
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    return model


def process_images_in_directory(directory_path, model_path, output_json_path):
    # 设置图像size
    image_size = 384
    # 加载 model
    model = load_blip_model(model_path, image_size=image_size, device=device)

    output_data = []
    # 对目录下的文件名进行排序，遍历的时候会按文件名顺序进行遍历
    filenames = sorted(os.listdir(directory_path))
    for filename in filenames:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)

            with torch.no_grad():
                image = load_demo_image(image_path, image_size=image_size, device=device)

                # beam search
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
                # nucleus sampling
                # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

                result = {
                    'id': len(output_data) + 1,  # Assign a unique ID
                    'filename': filename,
                    'caption': caption[0]
                }
                print(result)

                output_data.append(result)

        # Write results to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Process images and generate captions')
    parser.add_argument('--image_dir', default='./demo', type=str, help='Path to the directory containing images')
    parser.add_argument('--model_dir', default='./model_base_caption_capfilt_large.pth', type=str, help='Path to the directory containing pre-train model')
    parser.add_argument('--output_json_dir', default='./output', type=str, help='Path to the directory for JSON output')
    args = parser.parse_args()

    # 参数赋值
    images_dir = args.image_dir
    model_dir = args.model_dir
    output_json_file = os.path.join(args.output_json_dir, 'output.json')

    # 实现image-to-text的转换
    process_images_in_directory(images_dir, model_dir, output_json_file)


if __name__ == "__main__":
    main()
