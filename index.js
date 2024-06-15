import { AutoProcessor, RawImage, AutoModel, env } from 'https://jsd.onmicrosoft.cn/npm/@xenova/transformers';
env.allowLocalModels = false;
const status = document.getElementById('status');
const fileSelect = document.getElementById('file-select');
const imageContainer = document.getElementById('image-container');
const outputContainer = document.getElementById('output-container');

status.textContent = '加载模型中...';
let startTime = null;
let endTime = null;

// Load model and processor
const model = await AutoModel.from_pretrained('Xenova/modnet-onnx', { quantized: false });
const processor = await AutoProcessor.from_pretrained('Xenova/modnet-onnx');

status.textContent = 'Ready';
// Load image from URL
const url = 'demo.webp';
function useRemoteImage(url) {
  const image = document.createElement('img');
  image.crossOrigin = "anonymous";
  image.src = url;
  imageContainer.appendChild(image);
  setTimeout(() => start(url), 0)
}
useRemoteImage(url)

fileSelect.addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  // Set up a callback when the file is loaded
  reader.onload = function (e2) {
    status.textContent = 'Image loaded';

    imageContainer.innerHTML = '';
    outputContainer.innerHTML = '';
    const image = document.createElement('img');
    image.src = e2.target.result;
    imageContainer.appendChild(image);
    setTimeout(() => start(image.src), 0)
  };
  reader.readAsDataURL(file);
});

async function start(source) {
  startTime = new Date();
  status.textContent = '正在施法...';
  console.log('start process')

  const image = await RawImage.read(source);
  // Process image
  const { pixel_values: input } = await processor(image);

  // Predict alpha matte
  const { output } = await model({ input });
  console.log('image', RawImage)

  // Convert output tensor to RawImage
  const matteImage = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);

  console.log('matteImage', matteImage, output)

  async function renderRawImage(image) {
    let rawCanvas = await image.toCanvas();
    const canvas = document.createElement('canvas');
    outputContainer.appendChild(canvas); // 将新创建的 Canvas 添加到页面中
    canvas.width = image.width;
    canvas.height = image.height;

    const ctx = canvas.getContext('2d');

    ctx.drawImage(rawCanvas, 0, 0);

  }

  // renderRawImage(matteImage)

  async function getForeground(rawImage, maskImage) {
    const rawCanvas = rawImage.toCanvas();
    const rawCtx = rawCanvas.getContext('2d');

    const maskCanvas = maskImage.toCanvas();
    const maskCtx = maskCanvas.getContext('2d');

    const rawImageData = rawCtx.getImageData(0, 0, rawCanvas.width, rawCanvas.height);
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

    for (let i = 0; i < rawImageData.data.length; i += 4) {
      // 把灰度通道值（RGB 都一样，这里取 R），赋到原图的透明通道（每个像素的第 4 个值）
      rawImageData.data[i + 3] = maskImageData.data[i];
    }

    rawCtx.putImageData(rawImageData, 0, 0);
    return rawCanvas;
  }

  let foregroundCanvas = await getForeground(image, matteImage);

  // 使用示例：
  console.log('debug', foregroundCanvas);
  // 模拟异步操作，确保在完成操作后才继续执行
  foregroundCanvas.convertToBlob()
    .then(function (blob) {
      // 创建图片
      let img = new Image();

      // 创建 blob URL 并设置为图片的 src
      img.src = URL.createObjectURL(blob);

      // 将图片添加到 body 中或者其他 HTML 元素
      outputContainer.appendChild(img);
      endTime = new Date();
      const diff = (endTime - startTime) / 1000
      setTimeout(() => status.textContent = '已完成，用时: ' + diff + 's', 0)
    })
    .catch(function (error) {
      // 捕获和处理 blob 创建过程中可能出现的错误
      console.error("Blob creation error: ", error);
    });
}
