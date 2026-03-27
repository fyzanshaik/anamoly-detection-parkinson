#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_log.h>
#include <LiquidCrystal_I2C.h>
#include <ESPAsyncWebServer.h>
#include "classifier_weights.h"

#define LCD_SDA      21
#define LCD_SCL      22
#define FEATURE_DIM  10
#define NUM_CLASSES  4

const char* AP_SSID = "AnomalyNet";
const char* AP_PASS = "anomaly123";

uint8_t node1MAC[] = {0xC0, 0xCD, 0xD6, 0xCE, 0x4C, 0xD4};
uint8_t node2MAC[] = {0xC0, 0xCD, 0xD6, 0x8D, 0x5E, 0xFC};

typedef struct {
  uint8_t  nodeId;
  uint8_t  phase;
  uint8_t  dataQuality;
  bool     isAnomalous;
  float    anomalyScore;
  float    vibrationLevel;
  float    features[FEATURE_DIM];
} SensorPacket;

typedef struct {
  uint8_t targetNodeId;
  uint8_t faultType;
  uint8_t motorSpeed;
  uint8_t forcePhase;
} GatewayCommand;

struct NodeState {
  uint8_t  phase;
  uint8_t  dataQuality;
  bool     anomalous;
  float    score;
  float    vibration;
  float    features[FEATURE_DIM];
  int      faultNeural;
  int      faultRule;
  int      faultFinal;
  String   confidence;
  String   explanation;
  String   action;
  unsigned long lastSeen;
  bool     online;
} nodes[2];

const float GW_BASELINE[FEATURE_DIM] = {
  -9.15f, 0.55f, 2.35f, 0.06f, 0.07f, 0.08f, 9.47f, 0.05f, 9.55f, 9.47f
};

LiquidCrystal_I2C lcd(0x27, 16, 2);
AsyncWebServer    server(80);
AsyncEventSource  events("/events");

const char* FAULT_NAMES[] = {"NORMAL", "IMBALANCE", "BEARING", "LOOSENESS"};
const char* PHASE_NAMES[] = {"NORMAL", "RAMP", "ANOMALY", "RESOLVE"};

float relu(float x) { return x > 0 ? x : 0; }

int neuralClassify(float* feat) {
  float x[FEATURE_DIM];
  for (int i = 0; i < FEATURE_DIM; i++)
    x[i] = (feat[i] - cls_norm_mean[i]) / (cls_norm_std[i] + 1e-6f);

  float h1[8];
  for (int j = 0; j < 8; j++) {
    h1[j] = cls_b1[j];
    for (int i = 0; i < FEATURE_DIM; i++) h1[j] += x[i] * cls_w1[i][j];
    h1[j] = relu(h1[j]);
  }

  float h2[4];
  for (int j = 0; j < 4; j++) {
    h2[j] = cls_b2[j];
    for (int i = 0; i < 8; i++) h2[j] += h1[i] * cls_w2[i][j];
    h2[j] = relu(h2[j]);
  }

  float out[4];
  for (int j = 0; j < 4; j++) {
    out[j] = cls_b3[j];
    for (int i = 0; i < 4; i++) out[j] += h2[i] * cls_w3[i][j];
  }

  int best = 0;
  for (int i = 1; i < 4; i++) if (out[i] > out[best]) best = i;
  return best;
}

int ruleClassify(float* feat) {
  float dm_mean = feat[6] - GW_BASELINE[6];
  float dm_std  = feat[7] - GW_BASELINE[7];
  float dm_max  = feat[8] - GW_BASELINE[8];
  float d_std   = ((feat[3]-GW_BASELINE[3])+(feat[4]-GW_BASELINE[4])+(feat[5]-GW_BASELINE[5]))/3.0f;
  float impulse = feat[8] / (feat[6] + 0.01f);
  float variab  = feat[7] / (feat[6] + 0.01f);

  if (impulse > 1.08f && dm_max > dm_mean * 1.5f) return 2;
  if (variab  > 0.035f && dm_std > dm_mean * 0.5f) return 3;
  if (dm_mean > 0.15f  && d_std > 0.02f)            return 1;
  return 0;
}

void buildExplanation(NodeState& n) {
  float dm_max  = n.features[8] - GW_BASELINE[8];
  float dm_mean = n.features[6] - GW_BASELINE[6];
  float dm_std  = n.features[7] - GW_BASELINE[7];
  float d_std   = ((n.features[3]-GW_BASELINE[3])+(n.features[4]-GW_BASELINE[4])+(n.features[5]-GW_BASELINE[5]))/3.0f;

  int f = n.faultFinal;
  if      (f == 1) { n.explanation = "All axes elevated proportionally. mag_mean +" + String(dm_mean,2) + " above baseline. Periodic vibration pattern detected."; n.action = "Inspect shaft balance. Check for debris on rotating mass."; }
  else if (f == 2) { n.explanation = "Impulse events detected. mag_max +" + String(dm_max,2) + " while mag_mean relatively stable. Impulse ratio: " + String(n.features[8]/n.features[6],2) + "x."; n.action = "Inspect bearing surface. Check lubrication and alignment."; }
  else if (f == 3) { n.explanation = "Irregular amplitude variation. mag_std +" + String(dm_std,2) + " indicating random pattern across all axes (" + String(d_std,3) + " avg std delta)."; n.action = "Check fasteners and mounting. Tighten all mechanical connections."; }
  else             { n.explanation = "Anomaly detected but pattern unclear. Elevated vibration without dominant signature."; n.action = "Monitor closely. Run diagnostic cycle."; }

  if (n.faultNeural == n.faultRule)           n.confidence = "HIGH";
  else if (n.faultNeural == 0 || n.faultRule == 0) n.confidence = "MEDIUM";
  else                                         n.confidence = "LOW";
}

void pushSSE(int idx) {
  NodeState& n = nodes[idx];
  bool mlHit = (n.phase == 0 && n.anomalous);
  String json = "{\"id\":" + String(idx+1)
    + ",\"phase\":" + n.phase
    + ",\"score\":" + String(n.score, 3)
    + ",\"anomalous\":" + (n.anomalous ? "true" : "false")
    + ",\"vib\":"  + String(n.vibration, 2)
    + ",\"fault\":" + n.faultFinal
    + ",\"neural\":" + n.faultNeural
    + ",\"rule\":"  + n.faultRule
    + ",\"confidence\":\"" + n.confidence + "\""
    + ",\"explanation\":\"" + n.explanation + "\""
    + ",\"action\":\"" + n.action + "\""
    + ",\"quality\":" + n.dataQuality
    + ",\"mlhit\":" + (mlHit ? "true" : "false")
    + ",\"lastseen\":" + n.lastSeen
    + ",\"online\":" + (n.online ? "true" : "false") + "}";
  events.send(json.c_str(), "message", millis());
}

void updateLCD() {
  static unsigned long last = 0;
  if (millis() - last < 800) return;
  last = millis();

  const char* FLT_SHORT[] = {"NRM", "IMB", "BRG", "LSN"};
  const char  QUAL_CHAR[]  = {'R', 'S', 'D'};

  char l0[17], l1[17];
  auto fmtNode = [&](int i, char* buf) {
    if (!nodes[i].online) {
      snprintf(buf, 17, "N%d:OFFLINE      ", i+1);
    } else {
      char qc = QUAL_CHAR[nodes[i].dataQuality < 3 ? nodes[i].dataQuality : 1];
      const char* flt = FLT_SHORT[nodes[i].faultFinal < 4 ? nodes[i].faultFinal : 0];
      snprintf(buf, 17, "N%d:%-3s%c%.2f %c   ",
        i+1, flt, nodes[i].anomalous ? '!' : ' ', nodes[i].score, qc);
    }
  };
  fmtNode(0, l0);
  fmtNode(1, l1);
  lcd.setCursor(0, 0); lcd.print(l0);
  lcd.setCursor(0, 1); lcd.print(l1);
}

void onRecv(const uint8_t* mac, const uint8_t* data, int len) {
  if (len != sizeof(SensorPacket)) return;
  SensorPacket pkt;
  memcpy(&pkt, data, sizeof(pkt));
  int idx = pkt.nodeId - 1;
  if (idx < 0 || idx > 1) return;

  NodeState& n = nodes[idx];
  n.phase     = pkt.phase;
  n.dataQuality = pkt.dataQuality;
  n.anomalous = pkt.isAnomalous;
  n.score     = pkt.anomalyScore;
  n.vibration = pkt.vibrationLevel;
  memcpy(n.features, pkt.features, FEATURE_DIM * sizeof(float));
  n.lastSeen  = millis();
  n.online    = true;

  if (n.anomalous) {
    n.faultNeural = neuralClassify(n.features);
    n.faultRule   = ruleClassify(n.features);
    n.faultFinal  = (n.faultNeural == n.faultRule) ? n.faultNeural :
                    (n.faultNeural > 0 ? n.faultNeural : n.faultRule);
    if (n.faultFinal == 0) n.faultFinal = 1;
    buildExplanation(n);
  } else {
    n.faultNeural = 0; n.faultRule = 0; n.faultFinal = 0;
    n.confidence  = "HIGH";
    n.explanation = "System operating normally.";
    n.action      = "No action required.";
  }

  pushSSE(idx);

  char logbuf[160];
  snprintf(logbuf, sizeof(logbuf),
    "{\"id\":%d,\"msg\":\"[%lus] ph:%d sc:%.2f %s flt:%s cf:%s\"}",
    pkt.nodeId, millis()/1000, pkt.phase, pkt.anomalyScore,
    n.anomalous ? "ANOMALY" : "normal",
    FAULT_NAMES[n.faultFinal], n.confidence.c_str());
  events.send(logbuf, "log", millis());

  Serial.printf("[GW] N%d phase:%d score:%.2f %s fault:%s conf:%s\n",
    pkt.nodeId, pkt.phase, pkt.anomalyScore,
    n.anomalous ? "ANOMALY" : "normal",
    FAULT_NAMES[n.faultFinal], n.confidence.c_str());
}

void onSent(const uint8_t* mac, esp_now_send_status_t s) {}

void initESPNow() {
  WiFi.mode(WIFI_AP_STA);
  esp_now_init();
  esp_now_register_recv_cb(onRecv);
  esp_now_register_send_cb(onSent);
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, node1MAC, 6);
  peer.channel = 0; peer.encrypt = false; peer.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&peer);
  memcpy(peer.peer_addr, node2MAC, 6);
  esp_now_add_peer(&peer);
}

static const char HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anomaly Detection</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0a0a0a;color:#e0e0e0;padding:14px;max-width:600px;margin:auto}
h1{color:#00ff88;font-size:1em;margin-bottom:14px;text-align:center;letter-spacing:2px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.card{background:#141414;border:1px solid #2a2a2a;border-radius:8px;padding:12px}
.card.ok{border-color:#00aa44}.card.bad{border-color:#ff4444;background:#140808}
.card.off{opacity:.4}.lbl{font-size:.65em;color:#666;text-transform:uppercase;margin-bottom:4px}
.st{font-size:1.1em;font-weight:bold;margin-bottom:4px}
.ok .st{color:#00ff88}.bad .st{color:#ff4444}.off .st{color:#555}
.sc{font-size:.75em;color:#888}
.box{background:#140808;border:1px solid #ff4444;border-radius:8px;padding:12px;margin-bottom:12px;display:none}
.box.on{display:block}.box h3{color:#ff4444;margin-bottom:6px;font-size:.95em}
.cf{font-size:.7em;color:#666;margin-bottom:8px}
.ex{font-size:.82em;line-height:1.6;margin-bottom:8px;color:#ccc}
.ac{color:#ffaa00;font-size:.82em}
.btns{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-top:10px}
btn,button{background:#1a1a1a;border:1px solid #333;color:#bbb;padding:7px 4px;border-radius:4px;font-size:.7em;cursor:pointer;font-family:monospace;width:100%}
button:hover{background:#222}.cl{border-color:#00aa44!important;color:#00ff88!important}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#555}
.ok .dot{background:#00ff88}.bad .dot{background:#ff4444}
.ls{font-size:.6em;color:#444;margin-top:2px}
.mlhit{font-size:.65em;color:#ff88ff;margin-top:2px;display:none}
.mlhit.on{display:block}
.model-panel{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:8px;padding:10px;margin-bottom:12px;font-size:.65em;color:#666}
.model-panel h4{color:#4488ff;margin-bottom:6px;font-size:.9em;letter-spacing:1px}
.model-panel .row{display:flex;justify-content:space-between;margin-bottom:3px}
.model-panel .val{color:#aaa}
.term-wrap{margin-top:12px}.term-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.term-hdr .lbl{margin:0}.term-clr{background:none;border:none;color:#444;font-size:.65em;cursor:pointer;font-family:monospace;padding:0}
.term-clr:hover{color:#888}.term-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.term{background:#050505;border:1px solid #1a1a1a;border-radius:6px;padding:8px;height:160px;overflow-y:auto;font-size:.62em;line-height:1.6}
.term .ok{color:#00aa55}.term .bad{color:#ff5555}.term .dim{color:#444}
</style></head><body>
<h1>&#11042; DISTRIBUTED ANOMALY DETECTION</h1>
<div class="grid">
<div class="card off" id="c1"><div class="lbl">Node 1</div>
<div class="st"><span class="dot"></span><span id="s1">OFFLINE</span></div>
<div class="sc" id="d1">—</div>
<div class="ls" id="ls1">—</div>
<div class="mlhit" id="ml1">&#9889; ML model triggered</div></div>
<div class="card off" id="c2"><div class="lbl">Node 2</div>
<div class="st"><span class="dot"></span><span id="s2">OFFLINE</span></div>
<div class="sc" id="d2">—</div>
<div class="ls" id="ls2">—</div>
<div class="mlhit" id="ml2">&#9889; ML model triggered</div></div></div>
<div class="model-panel">
<h4>MODEL INFO</h4>
<div class="row"><span>Autoencoder</span><span class="val">10 &rarr; 5 &rarr; 10 &nbsp;|&nbsp; ReLU &nbsp;|&nbsp; threshold 0.966</span></div>
<div class="row"><span>Classifier</span><span class="val">10 &rarr; 8 &rarr; 4 &rarr; 4 &nbsp;|&nbsp; ReLU&times;2 + Softmax &nbsp;|&nbsp; 97.9% acc</span></div>
<div class="row"><span>Total size</span><span class="val">1196 bytes &nbsp;|&nbsp; 299 parameters &nbsp;|&nbsp; pure numpy trained</span></div>
<div class="row"><span>Detection</span><span class="val">100% fault detection &nbsp;|&nbsp; 5.2% FPR &nbsp;|&nbsp; edge inference only</span></div>
</div>
<div class="box" id="fb1">
<h3 id="ft1">—</h3><div class="cf" id="fc1">—</div>
<div class="ex" id="fe1">—</div><div class="ac" id="fa1">—</div></div>
<div class="box" id="fb2">
<h3 id="ft2">—</h3><div class="cf" id="fc2">—</div>
<div class="ex" id="fe2">—</div><div class="ac" id="fa2">—</div></div>
<div class="btns">
<button onclick="cmd(1,1)">N1 Imbalance</button>
<button onclick="cmd(1,2)">N1 Bearing</button>
<button onclick="cmd(1,3)">N1 Looseness</button>
<button class="cl" onclick="cmd(1,0)">N1 Clear</button>
<button onclick="cmd(2,1)">N2 Imbalance</button>
<button onclick="cmd(2,2)">N2 Bearing</button>
<button onclick="cmd(2,3)">N2 Looseness</button>
<button class="cl" onclick="cmd(2,0)">N2 Clear</button></div>
<div style="display:flex;gap:8px;margin-bottom:10px">
<button style="flex:1" id="syncbtn" onclick="sync()">sync state</button>
<button style="flex:1" onclick="clearFaults()">clear faults</button>
</div>
<div class="term-wrap">
<div class="term-hdr"><div class="lbl">SERIAL LOG</div><button class="term-clr" onclick="clearLogs()">clear</button></div>
<div class="term-grid">
<div><div class="lbl" style="margin-bottom:4px;font-size:.6em">NODE 1</div><div class="term" id="t1"></div></div>
<div><div class="lbl" style="margin-bottom:4px;font-size:.6em">NODE 2</div><div class="term" id="t2"></div></div>
</div></div>
<script>
const FN=['NORMAL','IMBALANCE','BEARING','LOOSENESS'];
const PN=['NORMAL','RAMP','ANOMALY','RESOLVE'];
const QL=['MPU:real','MPU:synth','MPU:lost'];
const state={};
const recvAt={};

setInterval(()=>{
  [1,2].forEach(n=>{
    if(!recvAt[n])return;
    const s=Math.floor((Date.now()-recvAt[n])/1000);
    const el=document.getElementById('ls'+n);
    if(el)el.textContent=s<2?'just now':s+'s ago';
  });
},1000);

function setT(id,val){const e=document.getElementById(id);if(e&&e.textContent!==val)e.textContent=val;}
function setCls(id,val){const e=document.getElementById(id);if(e&&e.className!==val)e.className=val;}

const es=new EventSource('/events');
es.addEventListener('log',e=>{
  const d=JSON.parse(e.data);
  const el=document.getElementById('t'+d.id);
  if(!el)return;
  const ln=document.createElement('div');
  ln.className=d.msg.includes('ANOMALY')?'bad':'ok';
  ln.textContent=d.msg;
  el.appendChild(ln);
  if(el.children.length>60)el.removeChild(el.firstChild);
  if(el.scrollHeight-el.scrollTop<220)el.scrollTop=el.scrollHeight;
});
es.onmessage=e=>{
  const d=JSON.parse(e.data),n=d.id;
  const prev=state[n]||{};
  state[n]=d;
  recvAt[n]=Date.now();
  const cls=d.online?(d.anomalous?'bad':'ok'):'off';
  setCls('c'+n,'card '+cls);
  setT('s'+n,d.online?(d.anomalous?FN[d.fault]:'NORMAL'):'OFFLINE');
  setT('d'+n,d.online?'score:'+d.score.toFixed(2)+' | '+PN[d.phase]+' | '+QL[d.quality]:'—');
  setCls('ml'+n,'mlhit'+(d.mlhit?' on':''));
  if(d.anomalous&&d.fault>0){
    setCls('fb'+n,'box on');
    setT('ft'+n,'NODE '+n+': '+FN[d.fault]+' DETECTED');
    setT('fc'+n,'Confidence: '+d.confidence+' | Neural: '+FN[d.neural]+' | Rule: '+FN[d.rule]);
    setT('fe'+n,d.explanation);
    setT('fa'+n,'→ '+d.action);
  } else if(!d.anomalous&&prev.anomalous){
    setCls('fb'+n,'box');
  }
};
es.onerror=()=>{};
function cmd(node,fault){fetch('/cmd?node='+node+'&fault='+fault);}
function sync(){fetch('/sync').then(()=>{const b=document.getElementById('syncbtn');b.textContent='synced';setTimeout(()=>b.textContent='sync state',1500);});}
function clearFaults(){setCls('fb1','box');setCls('fb2','box');}
function clearLogs(){['t1','t2'].forEach(id=>{const e=document.getElementById(id);if(e)e.innerHTML='';})}
</script>
<div style="margin-top:10px;font-size:.6em;color:#444;line-height:1.6;border-top:1px solid #1a1a1a;padding-top:8px">
Note: Dashboard reflects last received packet per node. Data may be delayed due to ESP-NOW wireless latency and is not guaranteed to be real-time synchronized across nodes.
</div>
</body></html>
)rawliteral";

void setup() {
  Serial.begin(115200);
  delay(500);
  esp_log_level_set("*", ESP_LOG_NONE);

  memset(nodes, 0, sizeof(nodes));

  Wire.begin(LCD_SDA, LCD_SCL);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("GATEWAY BOOT    ");
  lcd.setCursor(0, 1); lcd.print("Initializing... ");

  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.printf("[GW] AP: %s  IP: %s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
  Serial.printf("[GW] MAC: %s\n", WiFi.macAddress().c_str());

  initESPNow();

  events.onConnect([](AsyncEventSourceClient* c) {
    for (int i = 0; i < 2; i++) pushSSE(i);
  });
  server.addHandler(&events);
  server.on("/", HTTP_GET, [](AsyncWebServerRequest* r) {
    r->send_P(200, "text/html", HTML);
  });
  server.on("/sync", HTTP_GET, [](AsyncWebServerRequest* r) {
    for (int i = 0; i < 2; i++) pushSSE(i);
    r->send(200, "text/plain", "ok");
  });
  server.on("/cmd", HTTP_GET, [](AsyncWebServerRequest* r) {
    int node  = r->hasParam("node")  ? r->getParam("node")->value().toInt()  : 0;
    int fault = r->hasParam("fault") ? r->getParam("fault")->value().toInt() : 0;
    if (node >= 1 && node <= 2) {
      GatewayCommand cmd = {(uint8_t)node, (uint8_t)fault, 0, 0};
      uint8_t* mac = (node == 2) ? node2MAC : node1MAC;
      esp_now_send(mac, (uint8_t*)&cmd, sizeof(cmd));
    }
    r->send(200, "text/plain", "ok");
  });
  server.begin();

  lcd.setCursor(0, 0); lcd.print("GATEWAY READY   ");
  lcd.setCursor(0, 1); lcd.print(WiFi.softAPIP().toString().c_str());

  Serial.println("[GW] Ready. Waiting for nodes...");
}

void loop() {
  for (int i = 0; i < 2; i++) {
    if (nodes[i].online && millis() - nodes[i].lastSeen > 5000) {
      nodes[i].online = false;
      pushSSE(i);
    }
  }
  updateLCD();
  delay(50);
}
