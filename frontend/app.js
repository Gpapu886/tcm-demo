/* 前端逻辑：上传 -> /api/analyze -> 渲染检测框 + 证候卡 */
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultCard = document.getElementById('resultCard');
const canvas = document.getElementById('detCanvas');
const ctx = canvas.getContext('2d');
const modeBadge = document.getElementById('modeBadge');
let currentImage = null;

// ---- 上传 ----
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#1e3c78'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = '#b9c6d9'; });
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.style.borderColor = '#b9c6d9';
  if (e.dataTransfer.files[0]) setImage(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setImage(fileInput.files[0]); });

function setImage(file) {
  currentImage = file;
  const url = URL.createObjectURL(file);
  preview.src = url; preview.hidden = false;
  document.getElementById('dropHint').style.display = 'none';
}

// ---- 分析 ----
analyzeBtn.addEventListener('click', async () => {
  analyzeBtn.disabled = true; analyzeBtn.textContent = '分析中…';
  try {
    const fd = new FormData();
    if (currentImage) fd.append('image', currentImage);
    fd.append('text', textInput.value.trim());
    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await resp.json();
    render(data);
  } catch (err) {
    alert('请求失败: ' + err.message);
  } finally {
    analyzeBtn.disabled = false; analyzeBtn.textContent = '开始辨证分析';
  }
});

// ---- 渲染 ----
function render(d) {
  resultCard.hidden = false;
  modeBadge.textContent = d.mode === 'live' ? '● 真实推理' : '● 演示模式';
  modeBadge.className = 'mode-badge ' + (d.mode === 'live' ? 'live' : '');

  // 证候
  document.getElementById('finalTop1').textContent = d.final_top1 || '—';
  document.getElementById('confBar').style.width = (d.conf * 100) + '%';
  document.getElementById('confVal').textContent = (d.conf * 100).toFixed(1) + '%';
  document.getElementById('uAlea').textContent = d.u_alea;
  const adv = document.getElementById('advice');
  adv.textContent = d.advice_cn; adv.className = 'advice ' + d.advice;

  // top3
  const ul = document.getElementById('top3List');
  ul.innerHTML = '';
  (d.text_top3 || []).forEach((t, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<b>${t.syndrome}</b> <span class="reason">${t.reason || ''}</span>`;
    ul.appendChild(li);
  });

  // 属性 chips
  const chips = document.getElementById('attrList');
  chips.innerHTML = '';
  const probs = d.visual.probs || {};
  Object.entries(probs).filter(([, v]) => v > 0.15).forEach(([k, v]) => {
    const li = document.createElement('li');
    li.textContent = ATTR_CN[k] + ' ' + (v * 100).toFixed(0) + '%';
    chips.appendChild(li);
  });
  const attrText = document.getElementById('attrText');
  attrText.textContent = Object.entries(probs).filter(([, v]) => v > 0.15)
    .map(([k, v]) => ATTR_CN[k] + '(' + (v * 100).toFixed(0) + '%)').join('、') || '未检出明显属性';

  // 检测框
  drawBoxes(d);
}

const ATTR_CN = {
  hongshe: '红舌', zishe: '紫暗舌', pangdashe: '胖大舌', shoushe: '瘦薄舌',
  hongdianshe: '红点舌', liewenshe: '裂纹舌', chihenshe: '齿痕舌',
  baitaishe: '白苔', huangtaishe: '黄苔'
};
const ATTR_COLOR = {
  hongshe: '#e74c3c', zishe: '#8e44ad', pangdashe: '#2980b9', shoushe: '#16a085',
  hongdianshe: '#c0392b', liewenshe: '#2c3e50', chihenshe: '#d35400',
  baitaishe: '#95a5a6', huangtaishe: '#f1c40f'
};

function drawBoxes(d) {
  const img = new Image();
  img.onload = () => {
    // canvas 自适应原图比例
    const scale = Math.min(480 / img.width, 480 / img.height);
    canvas.width = img.width * scale; canvas.height = img.height * scale;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    (d.visual.boxes || []).forEach(b => {
      const x = b.x1 * scale, y = b.y1 * scale, w = (b.x2 - b.x1) * scale, h = (b.y2 - b.y1) * scale;
      ctx.strokeStyle = ATTR_COLOR[b.attr] || '#1e3c78';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillRect(x, y - 22, w + 4, 22);
      ctx.fillStyle = '#fff';
      ctx.font = '13px sans-serif';
      ctx.fillText(`${b.attr_cn} ${(b.conf * 100).toFixed(0)}%`, x + 2, y - 6);
    });
  };
  if (currentImage) {
    img.src = URL.createObjectURL(currentImage);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#93a2b8'; ctx.font = '14px sans-serif';
    ctx.fillText('未上传舌象图片', 20, 30);
  }
}
