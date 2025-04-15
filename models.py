import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# 移除XMem相关导入，添加注释说明
# sys.path.append("./XMem/")
# from XMem.inference.inference_core import InferenceCore
# from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
# 注意：如果需要完全恢复XMem功能，请取消上述注释

def get_langsam_output(image, model, segmentation_texts, segmentation_count):

    segmentation_texts = " . ".join(segmentation_texts)

    masks, boxes, phrases, logits = model.predict(image, segmentation_texts)

    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)

    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        box = box.unsqueeze(dim=0)
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})

    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()

    masks = masks.float()

    return masks, boxes, phrases



def get_chatgpt_output(client, model, new_prompt, messages, role, file=sys.stdout):

    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})

    return messages



def get_xmem_output(model, device, trajectory_length):
    """
    简化版的get_xmem_output函数，不依赖XMem模型
    返回一个简单的掩码列表，而不是进行实际的视觉追踪
    """
    # 创建一个空的掩码列表
    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    num_objects = len(np.unique(mask)) - 1
    
    # 为每个轨迹点创建假掩码
    masks = []
    for i in range(0, trajectory_length + 1, config.xmem_output_every):
        # 使用初始掩码，假设物体位置不变
        masks.append(mask)
        
        # 可选：创建可视化输出
        frame = np.array(Image.open(config.rgb_image_trajectory_path.format(step=i)).convert("RGB"))
        output = Image.fromarray(frame)  # 简单地使用原始图像
        output.save(config.xmem_output_path.format(step=i))
    
    return masks
