const form = document.querySelector('#request-form');
const requestField = document.querySelector('#user-request');
const profileField = document.querySelector('#profile');
const submitButton = document.querySelector('#submit-button');
const idleState = document.querySelector('#idle-state');
const resultState = document.querySelector('#result-state');
const errorState = document.querySelector('#error-state');
const confirmPanel = document.querySelector('#confirm-panel');
const confirmButton = document.querySelector('#confirm-button');
let confirmationToken = null;

function pretty(value) { return JSON.stringify(value, null, 2); }
function displayError(message) { errorState.textContent = message; errorState.hidden = false; }
function actionName(action) { return ({call:'调用工具',clarify:'需要澄清',refuse:'拒绝执行'})[action] || '无有效决策'; }
function policyName(code) { return ({confirmation_bypass:'写入确认不能跳过',record_mutation:'记录工具没有修改权限',unavailable_export:'当前没有导出工具',unsupported_operation:'统计方式不受支持',unsupported_dataset:'数据集不受支持',unsupported_record_type:'记录类型不受支持',missing_to_unit:'缺少目标单位',missing_sort_order:'缺少排序方向',missing_dataset:'缺少数据集',missing_operation:'缺少统计方式',missing_event_end:'缺少日程结束时间',missing_availability_range:'缺少查询起止时间',missing_record_id:'缺少记录编号',missing_expression:'缺少计算算式',missing_query:'缺少检索目标'})[code] || '请求需进一步核对'; }
function gateText(gate) {
  if (gate.result?.status === 'pending_confirmation') return '日程草案待确认，尚未写入';
  if (gate.result?.status === 'ok') return '工具已执行并返回结果';
  if (gate.stage === 'not_executed') return gate.assistant_text || '本次不调用工具';
  if (gate.gate_code) return `服务端已拦截：${gate.reasons?.[0] || gate.gate_code}`;
  return gate.result?.message || '请查看校验详情';
}

async function apiPost(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `请求失败（${response.status}）`);
  return payload;
}

function showResult(payload) {
  idleState.hidden = true;
  resultState.hidden = false;
  errorState.hidden = true;
  const gate = payload.gate;
  const badge = document.querySelector('#result-badge');
  badge.className = 'badge ' + (gate.result?.status === 'pending_confirmation' ? 'warning' : gate.gate_code && gate.stage !== 'not_executed' ? 'error' : 'success');
  badge.textContent = gate.result?.status === 'pending_confirmation' ? '待确认' : gate.gate_code && gate.stage !== 'not_executed' ? '已拦截' : '已完成';
  document.querySelector('#latency').textContent = `模型耗时 ${payload.latency_seconds} 秒`;
  document.querySelector('#model-json').textContent = payload.model_decision ? pretty(payload.model_decision) : payload.raw_model_output || '模型未输出可解析的 JSON';
  document.querySelector('#decision-summary').textContent = actionName(payload.decision?.action) + (payload.decision?.tool ? ` · ${payload.decision.tool}` : '');
  document.querySelector('#policy-summary').textContent = payload.policy_code ? `服务端已修正：${policyName(payload.policy_code)}` : '沿用模型建议';
  document.querySelector('#decision-json').textContent = payload.decision ? pretty(payload.decision) : '无有效系统决策';
  document.querySelector('#gate-summary').textContent = gateText(gate);
  document.querySelector('#gate-json').textContent = pretty(gate);
  confirmationToken = payload.confirmation_token;
  confirmPanel.hidden = !confirmationToken;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorState.hidden = true;
  submitButton.disabled = true;
  submitButton.textContent = '正在生成…';
  confirmationToken = null;
  confirmPanel.hidden = true;
  try {
    showResult(await apiPost('/decide', {user_request:requestField.value.trim(), profile:profileField.value}));
  } catch (error) {
    displayError(error.message || '请求失败，请检查本地服务是否仍在运行。');
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = '生成决策 <span aria-hidden="true">↗</span>';
  }
});

confirmButton.addEventListener('click', async () => {
  if (!confirmationToken) return;
  confirmButton.disabled = true;
  confirmButton.textContent = '正在确认…';
  try {
    const payload = await apiPost('/confirm', {confirmation_token:confirmationToken});
    confirmationToken = null;
    confirmPanel.hidden = true;
    const gate = payload.gate;
    const badge = document.querySelector('#result-badge');
    badge.className = 'badge ' + (gate.result?.status === 'ok' ? 'success' : 'error');
    badge.textContent = gate.result?.status === 'ok' ? '已创建' : '未创建';
    document.querySelector('#gate-summary').textContent = gateText(gate);
    document.querySelector('#gate-json').textContent = pretty(gate);
  } catch (error) {
    displayError(error.message || '确认失败');
  } finally {
    confirmButton.disabled = false;
    confirmButton.textContent = '确认创建日程';
  }
});

document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  const type = button.dataset.example;
  if (type === 'query') { requestField.value = '帮我查一下差旅报销制度'; profileField.value = 'query'; }
  if (type === 'calculate') { requestField.value = '帮我算一下 125.5 加 36.8 等于多少'; profileField.value = 'calculate'; }
  if (type === 'calendar') {
    const day = new Date(); day.setDate(day.getDate() + 7);
    requestField.value = `请在${day.getFullYear()}年${day.getMonth() + 1}月${day.getDate()}日10:00到11:00创建主题为团队复盘的日程`;
    profileField.value = 'calendar';
  }
  requestField.focus();
}));

fetch('/health').then(response => response.json()).then(data => {
  document.querySelector('.status-dot').classList.add('ready');
  document.querySelector('#model-status').textContent = `本地模型已就绪 · ${data.model}`;
}).catch(() => { document.querySelector('#model-status').textContent = '本地服务暂不可用'; });
