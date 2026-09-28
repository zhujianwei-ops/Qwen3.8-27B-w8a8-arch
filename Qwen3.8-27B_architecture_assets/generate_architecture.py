#!/usr/bin/env python3
"""Read local checkpoint metadata and draw a bilingual, reproducible architecture diagram.
No model weights are materialized; only safetensors JSON headers are read.
Rendering dependencies: Pillow. Vector output is native SVG with an embedded CJK font.
"""
from pathlib import Path
import base64, collections, csv, hashlib, html, json, math, struct, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/zhujianwei')
ASSETS = ROOT / 'Qwen3.8-27B_architecture_assets'
MODEL = Path('/home1/model/Qwen3.8-27B-w8a8')
STEM = 'Qwen3.8-27B_architecture'
HF = Path('/usr/local/python3.12.13/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
SG = ROOT / 'task01/sglang/python/sglang/srt'
FONT = Path('/home1/jiangzhilin/.trae-cn-server/bin/stable-6d9ed14f693545a76d84c8e25a6cdb67fa9a9a4d-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf')
if not FONT.exists():
    raise SystemExit('Please set FONT to a Chinese TrueType font in this script.')
ASSETS.mkdir(exist_ok=True)
cfg = json.loads((MODEL / 'config.json').read_text())
tc, vc = cfg['text_config'], cfg['vision_config']
quant = json.loads((MODEL / 'quant_model_description.json').read_text())
index = json.loads((MODEL / 'quant_model_weights.safetensors.index.json').read_text())
meta = {}
header_hashes = {}
for shard in sorted(set(index['weight_map'].values())):
    with (MODEL / shard).open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        raw = f.read(length)
    header_hashes[shard] = hashlib.sha256(raw).hexdigest()
    for name, v in json.loads(raw).items():
        if name != '__metadata__':
            assert name not in meta, name
            meta[name] = dict(v, shard=shard, quantization=quant.get(name, 'UNSPECIFIED'))
assert set(meta) == set(index['weight_map'])
for name, v in meta.items():
    assert v['shard'] == index['weight_map'][name]

def check(key, shape, dtype=None):
    assert meta[key]['shape'] == shape, (key, meta[key]['shape'], shape)
    if dtype:
        assert meta[key]['dtype'] == dtype, (key, meta[key]['dtype'], dtype)

assert tc['num_hidden_layers'] == 64
assert tc['layer_types'] == (['linear_attention'] * 3 + ['full_attention']) * 16
assert tc['hidden_size'] == 5120 and tc['intermediate_size'] == 17408
assert tc['num_attention_heads'] == 24 and tc['num_key_value_heads'] == 4 and tc['head_dim'] == 256
assert tc['linear_num_key_heads'] == 16 and tc['linear_num_value_heads'] == 48
assert tc['linear_key_head_dim'] == tc['linear_value_head_dim'] == 128
assert tc['output_gate_type'] == 'swish' and tc['attn_output_gate'] is True
assert tc['mtp_num_hidden_layers'] == 1 and not tc['mtp_use_dedicated_embeddings']
assert not tc['tie_word_embeddings'] and not cfg['tie_word_embeddings']
assert vc['depth'] == 27 and vc['hidden_size'] == 1152 and vc['intermediate_size'] == 4304
assert vc['num_heads'] == 16 and vc['out_hidden_size'] == 5120
assert not vc['deepstack_visual_indexes']
assert not any('.experts.' in k for k in meta)
check('model.language_model.embed_tokens.weight', [248320, 5120], 'BF16')
check('lm_head.weight', [248320, 5120], 'BF16')
check('model.language_model.norm.weight', [5120], 'BF16')
for i, kind in enumerate(tc['layer_types']):
    p = f'model.language_model.layers.{i}.'
    for name, shape in [('gate_proj', [17408,5120]), ('up_proj', [17408,5120]), ('down_proj',[5120,17408])]:
        check(p+'mlp.'+name+'.weight', shape, 'I8')
    for name in ['input_layernorm','post_attention_layernorm']:
        check(p+name+'.weight',[5120],'BF16')
    if kind == 'linear_attention':
        for name, shape, dtype in [('in_proj_qkv',[10240,5120],'I8'),('in_proj_z',[6144,5120],'I8'),
                                  ('in_proj_a',[48,5120],'BF16'),('in_proj_b',[48,5120],'BF16'),
                                  ('conv1d',[10240,1,4],'BF16'),('norm',[128],'BF16'),
                                  ('out_proj',[5120,6144],'BF16')]:
            check(p+'linear_attn.'+name+'.weight',shape,dtype)
        check(p+'linear_attn.A_log',[48],'BF16')
        check(p+'linear_attn.dt_bias',[48],'BF16')
    else:
        for name, shape in [('q_proj',[12288,5120]),('k_proj',[1024,5120]),('v_proj',[1024,5120]),('o_proj',[5120,6144])]:
            check(p+'self_attn.'+name+'.weight',shape,'I8')
        for name in ['q_norm','k_norm']:
            check(p+'self_attn.'+name+'.weight',[256],'BF16')
for i in range(27):
    p=f'model.visual.blocks.{i}.'
    for name, shape, dtype in [('attn.qkv',[3456,1152],'I8'),('attn.proj',[1152,1152],'I8'),
                             ('mlp.linear_fc1',[4304,1152],'I8'),('mlp.linear_fc2',[1152,4304],'BF16')]:
        check(p+name+'.weight',shape,dtype)
check('model.visual.patch_embed.proj.weight',[1152,3,2,16,16],'BF16')
check('model.visual.pos_embed.weight',[2304,1152],'BF16')
check('model.visual.merger.linear_fc1.weight',[4608,4608],'BF16')
check('model.visual.merger.linear_fc2.weight',[5120,4608],'BF16')
check('mtp.fc.weight',[5120,10240],'BF16')
for suffix in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','self_attn.q_norm','self_attn.k_norm','mlp.gate_proj','mlp.up_proj','mlp.down_proj','input_layernorm','post_attention_layernorm']:
    ref=meta['model.language_model.layers.3.'+suffix+'.weight']
    check('mtp.layers.0.'+suffix+'.weight',ref['shape'],ref['dtype'])
for suffix in ['norm','pre_fc_norm_hidden','pre_fc_norm_embedding']:
    check('mtp.'+suffix+'.weight',[5120],'BF16')
assert not any(k.startswith(('mtp.embed_tokens.','mtp.lm_head.')) for k in meta)
for k, v in meta.items():
    if k.endswith('.weight') and v['dtype']=='I8':
        assert v['quantization']=='W8A8_DYNAMIC'
        check(k[:-6]+'weight_scale',[v['shape'][0],1],'F32')
        check(k[:-6]+'weight_offset',[v['shape'][0],1],'F32')

def group(k):
    if k.startswith('model.visual.'): return 'vision'
    if k.startswith('model.language_model.layers.'): return 'decoder_layers'
    if k.startswith('model.language_model.embed_tokens.'): return 'token_embedding'
    if k.startswith('model.language_model.norm.'): return 'final_norm'
    if k.startswith('lm_head.'): return 'lm_head'
    if k.startswith('mtp.'): return 'mtp'
    return 'other'
counts, dtypes, modes = collections.Counter(), collections.Counter(), collections.Counter()
for k,v in meta.items():
    if k.endswith(('.weight_scale','.weight_offset')): continue
    n=math.prod(v['shape'])
    counts[group(k)]+=n; dtypes[v['dtype']]+=n; modes[v['quantization']]+=n
assert counts['other']==0
TOTAL=sum(counts.values()); MAIN=TOTAL-counts['mtp']
logical_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in meta.values())
assert logical_bytes==index['metadata']['total_size']
print('Verified:',len(meta),'tensors, 64 decoder layers, 27 vision blocks, 1 MTP layer.')
print('Model parameters excluding quant metadata:',dict(counts))

sources=[MODEL/'config.json',MODEL/'quant_model_weights.safetensors.index.json',MODEL/'quant_model_description.json',
         MODEL/'Qwen3.8-27B_best_practice.yaml',MODEL/'README.md',HF,
         SG/'models/qwen3_5.py',SG/'models/qwen3_5_mtp.py',SG/'configs/qwen3_next.py']
summary={
    'generated_date':'2026-09-22', 'checkpoint_directory':str(MODEL), 'architecture':cfg['architectures'][0],
    'text_config':tc,'vision_config':vc,'tensor_count':len(meta),'tensor_data_bytes':logical_bytes,
    'parameter_counts_excluding_quantization_metadata':dict(counts),
    'total_parameter_elements':TOTAL,'main_model_parameter_elements':MAIN,
    'parameter_elements_by_dtype':dict(dtypes),'parameter_elements_by_quantization':dict(modes),
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'safetensors_header_sha256':header_hashes,
    'verification':'All shard headers match index. Every decoder, vision and MTP projection checked against configuration and quantization metadata. No forward pass executed.',
    'gate_semantics':'output_gate_type=swish configures GatedDeltaNet output RMSNorm gate. Full attention output gate uses sigmoid; confirmed by local SGLang source.',
}
(ASSETS/'verified_metadata.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (ASSETS/'tensor_inventory.tsv').open('w') as f:
    writer=csv.writer(f,delimiter='\t'); writer.writerow(['name','shape','dtype','quantization','parameter_elements','shard'])
    for k,v in sorted(meta.items()):
        writer.writerow([k,str(v['shape']),v['dtype'],v['quantization'],math.prod(v['shape']),v['shard']])

# Native vector + raster scene; coordinate units below are SVG pixels.
W,H,SCALE=3200,3930,1.5
BG='#f4f7fb'; INK='#17283c'; MUTED='#526477'; LINE='#72859c'; BORDER='#ced9e6'
BLUE='#176fa6'; TEAL='#117f76'; PURPLE='#7253a3'; ORANGE='#a85c12'
BPALE='#edf6fc'; TPALE='#eaf7f3'; PPALE='#f4effb'; OPALE='#fff4e7'; WHITE='#ffffff'
class Scene:
    def __init__(self):
        self.svg=[]
        self.im=Image.new('RGB',(round(W*SCALE),round(H*SCALE)),BG)
        self.d=ImageDraw.Draw(self.im)
        self.fonts={}; self.small=[]
    def f(self,size):
        key=round(size*SCALE)
        if key not in self.fonts:self.fonts[key]=ImageFont.truetype(str(FONT),key)
        return self.fonts[key]
    def rect(self,x,y,w,h,fill=WHITE,stroke=BORDER,r=16,width=2):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in (x,y,x+w,y+h)),round(r*SCALE),fill,stroke,round(width*SCALE))
    def line(self,pts,color=LINE,width=2.6,dash=False,arrow=False):
        pts=[(float(x),float(y)) for x,y in pts]
        path=' '.join(f'{x},{y}' for x,y in pts)
        das=' stroke-dasharray="9 7"' if dash else ''
        self.svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{das}/>')
        pix=[(x*SCALE,y*SCALE) for x,y in pts]
        if not dash:self.d.line(pix,fill=color,width=round(width*SCALE),joint='curve')
        else:
            for (x1,y1),(x2,y2) in zip(pix,pix[1:]):
                length=math.hypot(x2-x1,y2-y1)
                if not length:continue
                for t in range(0,math.ceil(length),round(16*SCALE)):
                    end=min(t+9*SCALE,length)
                    self.d.line([(x1+(x2-x1)*t/length,y1+(y2-y1)*t/length),(x1+(x2-x1)*end/length,y1+(y2-y1)*end/length)],fill=color,width=round(width*SCALE))
        if arrow:
            (x1,y1),(x2,y2)=pts[-2:]; a=math.atan2(y2-y1,x2-x1)
            q=[(x2,y2),(x2-12*math.cos(a)+5.5*math.sin(a),y2-12*math.sin(a)-5.5*math.cos(a)),(x2-12*math.cos(a)-5.5*math.sin(a),y2-12*math.sin(a)+5.5*math.cos(a))]
            self.svg.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in q)+f'" fill="{color}"/>')
            self.d.polygon([(x*SCALE,y*SCALE) for x,y in q],fill=color)
    def text(self,x,y,text,size=25,color=INK,bold=False,anchor='start',max_width=None):
        text=str(text)
        if max_width:
            while self.d.textlength(text,font=self.f(size))>max_width*SCALE and size>16: size-=0.5
            assert self.d.textlength(text,font=self.f(size))<=max_width*SCALE+1,(text,max_width,size)
        if size<19:self.small.append((text,size))
        fw='700' if bold else '400'
        # Both renderers use an alphabetic baseline at y + font size.
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="ArchitectureCJK, sans-serif" font-size="{size}" font-weight="{fw}" fill="{color}" text-anchor="{anchor}">{html.escape(text)}</text>')
        a={'start':'ls','middle':'ms','end':'rs'}[anchor]
        self.d.text((x*SCALE,(y+size)*SCALE),text,font=self.f(size),fill=color,anchor=a,stroke_width=1 if bold and size>=26 else 0,stroke_fill=color)
        return size
    def lines(self,x,y,lines,size=25,color=INK,gap=34,anchor='start',max_width=None):
        for i,t in enumerate(lines):self.text(x,y+i*gap,t,size,color,anchor=anchor,max_width=max_width)
    def box(self,x,y,w,h,title,details=(),fill=WHITE,color=BLUE,size=27,badge=None):
        self.rect(x,y,w,h,fill,color,12,2)
        rows=[title]+list(details)
        gap=32 if h>=90 else 30
        total=size+(len(rows)-1)*gap
        yy=y+(h-total)/2-2
        self.text(x+w/2,yy,title,size,color,bold=True,anchor='middle',max_width=w-28)
        for j,t in enumerate(details):self.text(x+w/2,yy+(j+1)*gap,t,23,INK,anchor='middle',max_width=w-28)
        if badge:
            bw=75 if badge=='BF16' else 61
            self.rect(x+w-bw-10,y-12,bw,25,color,color,7,1)
            self.text(x+w-bw/2-10,y-10,badge,17,WHITE,bold=False,anchor='middle')
    def panel(self,x,y,w,h,label,title,subtitle,color):
        self.rect(x,y,w,h,WHITE,BORDER,22,2)
        self.rect(x+27,y+28,46,46,color,color,10,1)
        self.text(x+50,y+32,label,29,WHITE,True,'middle')
        self.text(x+89,y+30,title,31,INK,True,max_width=w-126)
        self.text(x+31,y+88,subtitle,23,MUTED,max_width=w-62)
    def circle(self,x,y,symbol='+',r=22,color=BLUE):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2.6"/>')
        self.d.ellipse(tuple(round(v*SCALE) for v in (x-r,y-r,x+r,y+r)),fill=WHITE,outline=color,width=4)
        self.text(x,y-19,symbol,31,color,True,'middle')
    def arrow(self,x1,y1,x2,y2,**kw):self.line([(x1,y1),(x2,y2)],arrow=True,**kw)

s=Scene()
s.text(62,47,'Qwen3.8-27B  模型结构细节图',61,INK,True,max_width=1900)
s.text(65,126,'本地检查点：Qwen3.8-27B-w8a8  |  架构类：Qwen3_5ForConditionalGeneration',29,MUTED,max_width=3000)
s.text(66,176,'维度来自配置与全部权重分片头；结构流程结合本地实现核对。图中层号从 1 开始，权重 layers 索引从 0 开始。',24,MUTED,max_width=3000)
# A: overview
s.panel(60,235,3080,650,'A','整体结构与 64 层排列','主干：文本与视觉特征合流 → 64 层 Dense Decoder → 最终 RMSNorm → 独立 LM Head',BLUE)
s.box(100,377,435,75,'文本 / 多模态 token IDs',('[B, T]（含图像 / 视频占位）',),BPALE,BLUE,size=27)
s.box(100,510,435,102,'Token Embedding',('权重 [248320, 5120]','输出 [B, T, 5120]'),BPALE,BLUE,badge='BF16')
s.arrow(317,452,317,510)
s.box(595,377,470,75,'图像 / 视频 RGB 输入',('共享同一个视觉编码器',),PPALE,PURPLE,size=27)
s.box(595,510,470,102,'Vision Encoder + Merger',('27 × VisionBlock；2 × 2 合并','视觉输出 [N / 4, 5120]'),PPALE,PURPLE)
s.arrow(830,452,830,510)
s.box(1140,465,465,148,'多模态序列融合',('用视觉特征替换占位 embedding','统一维度 [B, T, 5120]','构建文本 / 时空位置索引'),BPALE,BLUE)
s.line([(535,561),(562,561),(562,667),(1105,667),(1105,533),(1140,533)],arrow=True)
s.arrow(1065,561,1140,561,color=PURPLE)
s.rect(1690,390,880,240,TPALE,TEAL,18,2)
s.text(2130,413,'4 层一组 × 16 组 = 64 层',31,TEAL,True,'middle')
for i,txt in enumerate(['L1 · GDN','L2 · GDN','L3 · GDN','L4 · GQA']):
    xx=1720+i*210
    s.box(xx,478,185,72,txt,(),TPALE if i<3 else BPALE,TEAL if i<3 else BLUE,size=25)
    if i<3:s.arrow(xx+185,514,xx+207,514)
s.text(2130,575,'48 层线性注意力 + 16 层全注意力；每层均含 SwiGLU MLP',24,INK,anchor='middle',max_width=825)
s.arrow(1605,535,1690,535)
s.box(2670,376,420,77,'最终 RMSNorm',('hidden_size = 5120',),BPALE,BLUE)
s.line([(2570,515),(2617,515),(2617,414),(2670,414)],arrow=True)
s.box(2670,502,420,98,'LM Head · 不与 embedding 绑权',('Linear 5120 → 248320','权重 [248320, 5120]'),BPALE,BLUE,size=24,badge='BF16')
s.arrow(2880,453,2880,502)
s.box(2670,650,420,83,'输出 logits / 下一个 token',('[B, T, 248320]',),BPALE,BLUE,size=25)
s.arrow(2880,600,2880,650)
s.box(1710,709,860,83,'独立 MTP 辅助分支（见 F）',('主干隐藏状态 + 下一 token embedding → 1 层预测模块',),OPALE,ORANGE,size=27)
s.line([(2130,630),(2130,709)],color=ORANGE,dash=True,arrow=True)
s.lines(105,736,['D = 5120  |  FFN = 17408  |  词表 = 248320','配置最大位置数 = 262144（256K）；不是运行时容量保证'],25,MUTED,gap=38,max_width=1510)
s.text(2680,783,'主干 27.3567B 参数（含视觉）',24,MUTED,max_width=420)
# B: block / FFN
s.panel(60,930,1000,1350,'B','每个 Decoder Layer 与 SwiGLU','64 层共享该骨架；Token Mixer 由层类型选择 C 或 D。',BLUE)
s.box(250,1070,620,65,'输入 h  [B, T, 5120]',(),BPALE,BLUE)
s.box(250,1180,620,66,'input_layernorm · RMSNorm',('维度 5120；eps = 1e-6',),WHITE,BLUE,size=25)
s.arrow(560,1135,560,1180)
s.box(250,1290,620,89,'Token Mixer',('线性 Gated DeltaNet 或全注意力 GQA',),TPALE,TEAL)
s.arrow(560,1246,560,1290)
s.circle(560,1420,'+',22,BLUE); s.arrow(560,1379,560,1398)
s.line([(560,1155),(155,1155),(155,1420),(538,1420)],arrow=True)
s.text(160,1265,'残差',24,MUTED)
s.box(250,1480,620,67,'post_attention_layernorm',('RMSNorm(5120)，eps = 1e-6',),WHITE,BLUE,size=25)
s.arrow(560,1442,560,1480)
s.box(250,1590,620,65,'SwiGLU MLP · Dense',('5120 → 17408 → 5120',),BPALE,BLUE,size=26)
s.arrow(560,1547,560,1590)
s.circle(560,1700,'+',22,BLUE); s.arrow(560,1655,560,1678)
s.line([(560,1460),(930,1460),(930,1700),(582,1700)],arrow=True)
s.text(938,1540,'残差',22,MUTED)
s.arrow(560,1722,560,1767)
s.text(560,1776,'输出 h′  [B, T, 5120]',27,INK,True,'middle')
s.line([(95,1840),(1020,1840)],BORDER,2)
s.text(102,1860,'SwiGLU 内部：两路投影 → 门控乘法 → 下投影',26,INK,True,max_width=900)
s.box(115,1930,390,85,'gate_proj: 5120 → 17408',('接 SiLU / Swish 激活',),BPALE,BLUE,size=25,badge='W8A8')
s.box(610,1930,390,85,'up_proj: 5120 → 17408',('无激活的并行分支',),BPALE,BLUE,size=25,badge='W8A8')
s.text(560,1900,'同一输入 x',22,MUTED,anchor='middle')
s.line([(560,1928),(560,1917),(310,1917),(310,1930)],arrow=True)
s.line([(560,1917),(805,1917),(805,1930)],arrow=True)
s.circle(560,2075,'×',23,BLUE)
s.line([(310,2015),(310,2075),(537,2075)],arrow=True)
s.line([(805,2015),(805,2075),(583,2075)],arrow=True)
s.box(230,2130,660,69,'down_proj: 17408 → 5120',(),BPALE,BLUE,size=28,badge='W8A8')
s.arrow(560,2098,560,2130)
s.text(560,2220,'MLP(x) = W_down [ SiLU(W_gate x) × (W_up x) ]',24,INK,anchor='middle',max_width=890)
# C: Gated DeltaNet
s.panel(1100,930,1000,1350,'C','线性注意力 · Gated DeltaNet','共 48 层：每组的前 3 层；不形成 T × T 注意力矩阵。',TEAL)
s.box(1180,1070,840,65,'归一化后的输入 x  [B, T, 5120]',(),TPALE,TEAL)
s.line([(1600,1135),(1600,1160),(1350,1160),(1350,1190)],arrow=True)
s.line([(1600,1160),(1708,1160),(1708,1190)],arrow=True)
s.line([(1708,1160),(1948,1160),(1948,1190)],arrow=True)
s.box(1135,1190,425,92,'in_proj_qkv',('5120 → 10240',),TPALE,TEAL,badge='W8A8')
s.box(1580,1190,225,92,'in_proj_z',('5120 → 6144',),TPALE,TEAL,size=25,badge='W8A8')
s.box(1830,1190,235,92,'in_proj_a / b',('各 5120 → 48',),TPALE,TEAL,size=24,badge='BF16')
s.box(1135,1330,425,95,'因果 Depthwise Conv1D',('10240 通道，kernel = 4','权重 [10240, 1, 4]；接 SiLU'),TPALE,TEAL,size=25,badge='BF16')
s.arrow(1347,1282,1347,1330)
s.box(1135,1470,425,121,'拆分 Q / K / V',('Q、K：各 16 × 128 = 2048','V：48 × 128 = 6144','2048 + 2048 + 6144 = 10240'),TPALE,TEAL,size=27)
s.arrow(1347,1425,1347,1470)
s.box(1135,1640,425,91,'Q / K：L2 归一化',('各重复 3 份：16 → 48 个头',),TPALE,TEAL,size=25)
s.arrow(1347,1591,1347,1640)
s.box(1600,1370,440,173,'每个 value head 的更新门',('beta = sigmoid(b)','g = -exp(A_log) × softplus(a + dt_bias)','A_log / dt_bias：各 [48]'),TPALE,TEAL,size=26)
s.arrow(1948,1282,1948,1370)
s.box(1160,1790,790,145,'Gated Delta Rule · 因果状态更新',('prefill：分块计算；decode：递推更新','状态 S：[B, 48, 128, 128]','输出 [B, T, 48, 128]'),TPALE,TEAL,size=29)
s.arrow(1347,1731,1347,1790)
s.line([(1820,1543),(1820,1738),(1810,1738),(1810,1790)],arrow=True)
s.box(1185,1990,840,100,'Gated RMSNorm（按每头 128 维）',('RMSNorm(output) × Swish(z)','output_gate_type = swish；展平为 6144 维'),TPALE,TEAL,size=28,badge='BF16')
s.arrow(1555,1935,1555,1990)
s.line([(1692,1282),(1692,1312),(2070,1312),(2070,2040),(2025,2040)],color=TEAL,arrow=True)
s.text(2009,1760,'z',25,TEAL,True)
s.box(1260,2140,680,65,'out_proj: 6144 → 5120',(),TPALE,TEAL,size=29,badge='BF16')
s.arrow(1605,2090,1605,2140)
s.text(1600,2228,'保留卷积历史与递推状态；mamba_ssm_dtype 配置为 float32',23,MUTED,anchor='middle',max_width=920)
# D full attention
s.panel(2140,930,1000,1350,'D','全注意力 · Gated GQA','共 16 层：第 4、8、12、…、64 层（权重索引 3、7、…、63）。',BLUE)
s.box(2210,1070,860,65,'归一化后的输入 x  [B, T, 5120]',(),BPALE,BLUE)
s.line([(2640,1135),(2640,1160),(2385,1160),(2385,1190)],arrow=True)
s.line([(2640,1160),(2730,1160),(2730,1190)],arrow=True)
s.line([(2730,1160),(2980,1160),(2980,1190)],arrow=True)
s.box(2180,1190,410,86,'q_proj: 5120 → 12288',('联合投影 Q 与输出 gate',),BPALE,BLUE,size=25,badge='W8A8')
s.box(2620,1190,220,86,'k_proj',('5120 → 1024',),BPALE,BLUE,size=27,badge='W8A8')
s.box(2870,1190,220,86,'v_proj',('5120 → 1024',),BPALE,BLUE,size=27,badge='W8A8')
s.box(2180,1310,410,72,'按头拆分 Q / gate',('各 24 × 256 = 6144',),BPALE,BLUE,size=25)
s.arrow(2385,1276,2385,1310)
s.box(2250,1430,305,69,'Q · RMSNorm(256)',('Q：24 个头',),WHITE,BLUE,size=25)
s.arrow(2402,1382,2402,1430)
s.box(2620,1430,220,69,'K · RMSNorm',('4 头，每头 256',),WHITE,BLUE,size=23)
s.arrow(2730,1276,2730,1430)
s.box(2280,1550,565,108,'Q / K · 部分 M-RoPE',('每头旋转 64 / 256 维（25%）','theta = 10,000,000；sections = [11,11,10]'),BPALE,BLUE,size=27)
s.arrow(2402,1499,2402,1550);s.arrow(2730,1499,2730,1550)
s.box(2480,1730,610,145,'因果 GQA Attention',('24 Q heads / 4 KV heads = 6 : 1','softmax(QK^T / 16 + causal_mask) V','输出 [B, T, 24, 256] → 6144'),BPALE,BLUE,size=29)
s.line([(2562,1658),(2562,1690),(2690,1690),(2690,1730)],arrow=True)
s.line([(2980,1276),(3085,1276),(3085,1690),(2970,1690),(2970,1730)],arrow=True)
s.text(3010,1522,'V',24,BLUE,True)
s.box(2200,1932,280,76,'sigmoid(gate)',('维度 6144',),BPALE,BLUE,size=25)
s.line([(2180,1346),(2162,1346),(2162,1970),(2200,1970)],color=BLUE,arrow=True)
s.circle(2790,1970,'×',25,BLUE)
s.arrow(2785,1875,2785,1945);s.arrow(2480,1970,2765,1970)
s.box(2465,2070,625,70,'o_proj: 6144 → 5120',(),BPALE,BLUE,size=29,badge='W8A8')
s.arrow(2790,1995,2790,2070)
s.lines(2185,2170,['每层 KV cache：K、V 各 [B, 4, T, 256]','attention_bias = false；attention_dropout = 0'],24,MUTED,gap=37,max_width=910)
# E vision
s.panel(60,2340,1000,1320,'E','视觉编码器 · 27 层 ViT + Merger','图像 / 视频共享权重；视觉输出与语言维度对齐为 5120。',PURPLE)
s.box(110,2475,900,73,'RGB 图像 / 视频分块',('每个 patch 含 3 × 2 × 16 × 16 个输入元素',),PPALE,PURPLE,size=28)
s.box(110,2595,900,100,'Patch Embedding · Conv3D',('kernel / stride = (2, 16, 16)，3 → 1152','权重 [1152, 3, 2, 16, 16]；输出 [N, 1152]'),PPALE,PURPLE,size=29,badge='BF16')
s.arrow(560,2548,560,2595)
s.box(110,2740,900,90,'加可学习的位置嵌入（空间插值）',('pos_embed.weight [2304, 1152]；2304 = 48 × 48',),PPALE,PURPLE,size=28,badge='BF16')
s.arrow(560,2695,560,2740)
s.rect(110,2875,900,342,PPALE,PURPLE,15,2)
s.text(560,2897,'VisionBlock × 27（blocks.0 … blocks.26）',29,PURPLE,True,'middle')
s.box(145,2960,190,65,'LayerNorm',(),WHITE,PURPLE,size=24)
s.box(380,2960,505,65,'非因果 MHA：16 heads × 72',(),WHITE,PURPLE,size=24)
s.circle(958,2992,'+',21,PURPLE)
s.arrow(335,2992,380,2992);s.arrow(885,2992,937,2992)
s.text(565,3040,'QKV: 1152 → 3456；Q/K 空间 RoPE；proj: 1152 → 1152',22,PURPLE,anchor='middle',max_width=820)
s.box(145,3100,190,65,'LayerNorm',(),WHITE,PURPLE,size=24)
s.box(380,3100,505,65,'MLP: 1152 → 4304 → 1152',('GELU（tanh 近似）',),WHITE,PURPLE,size=24)
s.circle(958,3132,'+',21,PURPLE)
s.arrow(335,3132,380,3132);s.arrow(885,3132,937,3132)
s.line([(958,3013),(990,3013),(990,3078),(130,3078),(130,3132),(145,3132)],color=PURPLE,arrow=True)
s.line([(240,2939),(240,2928),(958,2928),(958,2971)],color=PURPLE,dash=True,arrow=True)
s.line([(145,3090),(145,3085),(958,3085),(958,3111)],color=PURPLE,dash=True,arrow=True)
s.arrow(560,2830,560,2875)
s.box(110,3265,900,192,'Patch Merger · 全部 BF16',('LayerNorm(1152) → 空间 2 × 2 合并','[N, 1152] → [N / 4, 4608]','FC 4608 → 4608 → GELU → FC 4608 → 5120','输出 token 数减少至 1/4；out_hidden_size = 5120'),PPALE,PURPLE,size=29)
s.arrow(560,3217,560,3265)
s.box(110,3500,900,77,'视觉 embedding  [N / 4, 5120]',('回到 A：替换图像 / 视频占位 token 的 embedding',),PPALE,PURPLE,size=28)
s.arrow(560,3457,560,3500)
s.text(110,3601,'N 为合并前 patch 数；deepstack_visual_indexes = []（无该分支）。',23,MUTED,max_width=910)
# F MTP
s.panel(1100,2340,1000,1320,'F','MTP · 1 层多词元预测辅助模块','权重确实包含 mtp.*；不计入主干的 64 层 Decoder。',ORANGE)
s.box(1140,2490,425,78,'主干隐藏状态 h',('[B, T, 5120]',),OPALE,ORANGE,size=27)
s.box(1630,2490,425,78,'下一 token 的 embedding e',('复用主干词嵌入；5120 维',),OPALE,ORANGE,size=25)
s.box(1140,2615,425,80,'pre_fc_norm_hidden',('RMSNorm(5120)',),OPALE,ORANGE,size=25,badge='BF16')
s.box(1630,2615,425,80,'pre_fc_norm_embedding',('RMSNorm(5120)',),OPALE,ORANGE,size=24,badge='BF16')
s.arrow(1352,2568,1352,2615);s.arrow(1842,2568,1842,2615)
s.box(1240,2745,720,70,'Concat [e, h] → 10240',(),OPALE,ORANGE,size=28)
s.line([(1352,2695),(1352,2720),(1500,2720),(1500,2745)],arrow=True)
s.line([(1842,2695),(1842,2720),(1700,2720),(1700,2745)],arrow=True)
s.box(1240,2860,720,75,'mtp.fc: 10240 → 5120',('权重 [5120, 10240]；无 bias',),OPALE,ORANGE,size=28,badge='BF16')
s.arrow(1600,2815,1600,2860)
s.box(1180,2980,840,115,'mtp.layers.0 · 1 个全注意力 Decoder',('Gated GQA（同 D） + SwiGLU（同 B）','24 Q heads / 4 KV heads；FFN = 17408'),OPALE,ORANGE,size=27)
s.arrow(1600,2935,1600,2980)
s.box(1240,3140,720,63,'mtp.norm · RMSNorm(5120)',(),OPALE,ORANGE,size=27,badge='BF16')
s.arrow(1600,3095,1600,3140)
s.box(1240,3247,720,71,'复用主干 LM Head: 5120 → 248320',(),OPALE,ORANGE,size=26,badge='BF16')
s.arrow(1600,3203,1600,3247)
s.box(1240,3363,720,71,'MTP 辅助 logits / 候选 token',(),OPALE,ORANGE,size=29)
s.arrow(1600,3318,1600,3363)
s.lines(1145,3477,['mtp_num_hidden_layers = 1','mtp_use_dedicated_embeddings = false','独立参数约 0.4247B；是否启用及调度方式由推理引擎决定。','普通自回归生成主路径仍为 A；MTP 是单独的辅助分支。'],24,MUTED,gap=36,max_width=910)
# G Quantization and parameter audit
s.panel(2140,2340,1000,1320,'G','实际量化范围与权重核验','W8A8_DYNAMIC：权重 INT8 / per-channel，激活 INT8 / per-token。',TEAL)
rows=[
 ('文本 / MTP：MLP gate、up、down','W8A8'),
 ('全注意力：q、k、v、o 投影','W8A8'),
 ('GDN：in_proj_qkv、in_proj_z','W8A8'),
 ('GDN：out、a/b、conv、norm','BF16'),
 ('VisionBlock：qkv、proj、fc1','W8A8'),
 ('VisionBlock fc2 / Patch / Merger','BF16'),
 ('Embedding / LM Head / Norm 权重','BF16'),
]
s.rect(2175,2490,930,58,TPALE,BORDER,9,1)
s.text(2193,2500,'模块 / 权重',25,TEAL,True)
s.text(3015,2500,'格式',25,TEAL,True,'middle')
for i,(mod,typ) in enumerate(rows):
    yy=2548+i*70
    s.rect(2175,yy,930,70,WHITE if i%2==0 else '#f7f9fc',BORDER,0,1)
    s.text(2193,yy+21,mod,25,INK,max_width=695)
    s.text(3015,yy+21,typ,25,TEAL if typ=='W8A8' else MUTED,True,'middle')
s.lines(2180,3060,['蓝绿色 W8A8 标记只说明相应投影的量化；不是整个模型全 INT8。','量化 scale / offset 为 FP32；视觉量化 Linear 的 bias 为 FP32。','图中 BF16 标记描述已保存权重；计算与缓存精度由实现决定。'],23,MUTED,gap=35,max_width=920)
s.rect(2175,3200,930,244,TPALE,TEAL,16,2)
s.text(2200,3223,'参数元素计数（剔除 weight_scale / weight_offset）',25,TEAL,True,max_width=880)
for yy,left,right in [(3275,'主干，含视觉 / embedding / LM Head','27.3567 B'),(3324,'独立 MTP 分支','0.4247 B'),(3373,'检查点全部模型参数','27.7814 B')]:
    s.text(2200,yy,left,25,INK,max_width=625)
    s.text(3064,yy,right,28,TEAL,True,'end')
s.lines(2180,3475,[f"INT8 参数：{dtypes['I8']/1e9:.4f} B；浮点参数：{(TOTAL-dtypes['I8'])/1e9:.4f} B",f"10 个权重分片 / {len(meta)} 个张量条目（含量化元数据）",f"张量数据合计：{logical_bytes/1e9:.3f} GB（不含运行时缓存）",'已逐层核验形状、dtype、索引及量化标记；未加载权重做推理。'],24,MUTED,gap=36,max_width=920)
# Footer provenance
s.line([(60,3705),(3140,3705)],BORDER,2)
s.text(65,3732,'数据来源与阅读说明',27,INK,True)
s.lines(65,3777,[
'主要依据：/home1/model/Qwen3.8-27B-w8a8/config.json、quant_model_description.json、量化 YAML、权重索引及全部 safetensors 分片头。',
'流程核对：本机 Transformers qwen3_5 与 SGLang qwen3_5 / qwen3_5_mtp；名称沿用目录，架构类以本地配置为准。',
'记号：B = batch；T = 语言 / 多模态序列长度；N = 合并前视觉 patch 数；Linear 用 输入维 → 输出维，权重矩阵按 [输出维, 输入维]。',
'生成日期：2026-09-22  |  SVG 内嵌中文字体，可无损放大；精确计数、完整张量清单与复现脚本见同名 assets 目录。'
],23,MUTED,gap=33,max_width=3060)

assert not [(t,z) for t,z in s.small if z<17], s.small
font64=base64.b64encode(FONT.read_bytes()).decode()
svg='<?xml version="1.0" encoding="UTF-8"?>\n'+f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">\n'
svg+='<title id="title">Qwen3.8-27B 本地 W8A8 检查点模型结构细节图</title>\n<desc id="desc">根据本地配置、所有权重张量元数据和本地源码核对。包含 64 层混合注意力文本解码器、27 层视觉编码器、MTP 和实际量化范围。</desc>\n'
svg+='<defs><style>@font-face { font-family: ArchitectureCJK; src: url(data:font/ttf;base64,'+font64+') format("truetype"); } text { font-kerning: normal; }</style></defs>\n'
svg+=f'<rect width="{W}" height="{H}" fill="{BG}"/>\n'+'\n'.join(s.svg)+'\n</svg>\n'
(ROOT/(STEM+'.svg')).write_text(svg)
s.im.save(ROOT/(STEM+'.png'),dpi=(180,180),optimize=True)
# PDF uses a raster rendition; the SVG is the lossless vector source.
s.im.save(ROOT/(STEM+'.pdf'),'PDF',resolution=180.0,title='Qwen3.8-27B architecture',author='Checkpoint metadata analysis')
preview=s.im.copy(); preview.thumbnail((1600,1965)); preview.save(ASSETS/'preview.png')
for key,bounds in {'A_overview':(60,235,3140,885),'B_decoder_mlp':(60,930,1060,2280),'C_gated_deltanet':(1100,930,2100,2280),'D_full_attention':(2140,930,3140,2280),'E_vision':(60,2340,1060,3660),'F_mtp':(1100,2340,2100,3660),'G_quantization':(2140,2340,3140,3660)}.items():
    crop=s.im.crop(tuple(round(v*SCALE) for v in bounds))
    crop.save(ASSETS/(key+'.png'),dpi=(180,180),optimize=True)

# Exact, readable supporting record; tables use checkpoint shapes rather than inference guesses.
notes=f'''# Qwen3.8-27B 本地模型结构说明

分析目录：`{MODEL}`。生成日期：2026-09-22。

主图：`{STEM}.svg`（矢量、内嵌中文字体）、`{STEM}.png`（4800 × 5895）和 `{STEM}.pdf`（PNG 的单页 PDF 版本）。七个模块的独立 PNG 在 `{ASSETS.name}/`。

## 依据与边界

模型名称沿用本地目录和 README。配置中的架构为 `Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，`language_model_only=false`，任务为 `image-text-to-text`。该记录描述指定的本地检查点，不以架构类中的 “3_5” 推断目录名称错误，也不把其他同名模型的网上参数替代为本地参数。

直接读取了 config、量化说明、量化 YAML、权重索引，以及全部 10 个 safetensors 文件的 JSON 头。共核对 {len(meta)} 个张量条目，包含 scale/offset。没有把约 32 GB 权重加载进内存，没有执行模型前向或精度测试。

模块数据流参考本机 Transformers 5.5.4 的 qwen3_5 实现，以及工作目录内 SGLang 的 qwen3_5 / qwen3_5_mtp 实现。模型目录未包含自己的 modeling 源码；配置记录的 transformers_version 为 5.8.0.dev0，而目录 README 的运行建议为 5.14.0。图示的形状和量化属性以实际检查点为准，具体融合算子、缓存布局和 MTP 调度由运行引擎决定。

## 1. 全局结构

`文本 IDs → Token Embedding` 与 `图像/视频 → Vision Encoder → Patch Merger` 在输入序列处合流：视觉特征替换图像/视频占位 token 的 embedding，而不是通过单独的 cross-attention 层注入。

随后是 `64 层混合注意力 Dense Decoder → RMSNorm → LM Head → logits`。

| 配置项 | 实际值 |
|---|---:|
| text hidden_size | 5120 |
| intermediate_size | 17408 |
| num_hidden_layers | 64 |
| vocab_size | 248320 |
| max_position_embeddings | 262144（256K，配置上限） |
| tie_word_embeddings | false |
| RMSNorm eps | 1e-6 |
| attention_bias / dropout | false / 0 |
| 架构形态 | Dense；权重中无 MoE experts/router |

主干层从 1 开始编号时：`[GDN, GDN, GDN, GQA] × 16`。全注意力在第 4、8、12、…、64 层；对应权重的零起始索引为 3、7、11、…、63。

Token Embedding 与 LM Head 分别有 `[248320,5120]` 的 BF16 权重，每个包含 1,271,398,400 个元素。两者不绑权。所有层的残差维度都是 5120；注意力内的 Q 或 V 总维度可以是 6144，因此不能把 `hidden_size / num_attention_heads` 当作这里的 head_dim。

## 2. Decoder 公共骨架与 FFN

每层为：

```text
h1 = h + TokenMixer(input_layernorm(h))
h2 = h1 + MLP(post_attention_layernorm(h1))
MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

`gate_proj` 和 `up_proj` 的权重都是 `[17408,5120]`；`down_proj` 是 `[5120,17408]`。主干 64 层和 MTP 1 层均为 W8A8_DYNAMIC。语言主干的普通 RMSNorm 实现使用 `(1 + weight)` 的缩放参数化；GDN 的 gated RMSNorm 使用其自身的缩放权重。它们不是视觉模块的 LayerNorm。

## 3. Gated DeltaNet 线性注意力（48 层）

| 模块 | 权重形状 | 保存 dtype / 量化 |
|---|---|---|
| in_proj_qkv | [10240,5120] | INT8 / W8A8_DYNAMIC |
| in_proj_z | [6144,5120] | INT8 / W8A8_DYNAMIC |
| in_proj_a / in_proj_b | 各 [48,5120] | BF16 / FLOAT |
| conv1d | [10240,1,4] | BF16 / FLOAT |
| A_log / dt_bias | 各 [48] | BF16 / FLOAT |
| norm.weight | [128] | BF16 / FLOAT |
| out_proj | [5120,6144] | BF16 / FLOAT |

QKV 经 10240 通道、kernel=4 的因果 depthwise Conv1D 和 SiLU 后拆分：Q=2048、K=2048、V=6144。Q/K 各 16 头、每头 128；V 为 48 头、每头 128。Q/K 经 L2 归一化，并按头重复三份以匹配 48 个 value heads。递推实现还按 `1/sqrt(128)` 缩放 Q。

每个 value head 有 `beta=sigmoid(b)` 与 `g=-exp(A_log)*softplus(a+dt_bias)`。用列向量表达，单步逻辑可以写为：

```text
S_decay = exp(g_t) * S_previous
error   = v_t - transpose(S_decay) @ k_t
S_t     = S_decay + beta_t * outer(k_t, error)
y_t     = transpose(S_t) @ q_t
```

S 的逻辑形状为 `[B,48,128,128]`；q 已包括归一化/缩放。prefill 可以使用分块并行实现，decode 用递推状态；它不保存每个历史 token 的传统 K/V 张量。还需要短卷积历史缓存，物理存储布局视引擎而定。配置 `mamba_ssm_dtype=float32` 说明递推状态的预期精度，不代表 checkpoint 的 `A_log` 是 FP32（实际为 BF16）。

最后：每头 128 维的 RMSNorm → 乘 `Swish(z)` → 拼为 6144 维 → `out_proj` 回到 5120。

**门控语义核对：`output_gate_type=swish` 控制 GDN 输出归一化门控；它不控制下面全注意力中的 sigmoid gate。** SGLang 的配置文档和两个模块实现均支持该区分。

## 4. Gated GQA 全注意力（16 层）

| 模块 | 权重形状 | dtype |
|---|---|---|
| q_proj（包含 Q 与 gate） | [12288,5120] | INT8 |
| k_proj | [1024,5120] | INT8 |
| v_proj | [1024,5120] | INT8 |
| o_proj | [5120,6144] | INT8 |
| q_norm / k_norm | 各 [256] | BF16 |

q_proj 输出按头组织为 `24 × (256 + 256)`，分别得到 Q 与输出 gate。二者各 6144 维；不能把 12288 都当作 Q。K/V 各 4 头、每头 256，Q:KV 头比为 6:1。Q/K 都先做逐头 RMSNorm，然后在每头前 64 维应用部分 M-RoPE，另外 192 维不旋转。

RoPE：`partial_rotary_factor=0.25`，`rope_theta=10000000`，`mrope_interleaved=true`，`mrope_section=[11,11,10]`（32 个频率对，对应 64 个旋转维度）。

```text
attention = softmax(QK^T / sqrt(256) + causal_mask) @ V
output    = o_proj(flatten(attention) * sigmoid(gate))
```

这里 `sqrt(256)=16`。每层 K/V cache 的逻辑形状分别为 `[B,4,T,256]`，无需把重复后的 24 个头都存入缓存。图示的张量布局是逻辑布局，不限定后端的物理布局。

## 5. 视觉模块

| 项目 | 实际结构 |
|---|---|
| 输入 | RGB 图像 / 视频，in_channels=3 |
| patch embedding | Conv3D，kernel=stride=(2,16,16)，3→1152，含 bias |
| patch 权重 | [1152,3,2,16,16]，BF16 |
| 位置嵌入 | [2304,1152]，空间网格 48×48，插值到输入网格 |
| 视觉深度 | 27 个 VisionBlock |
| 注意力 | 非因果 MHA，16 heads × 72；Q/K 使用空间 RoPE |
| QKV / proj 权重 | [3456,1152] / [1152,1152]，INT8 |
| FFN | 1152→4304→1152，GELU(tanh approximation) |
| FFN fc1 / fc2 权重 | [4304,1152] INT8 / [1152,4304] BF16 |
| 归一化 | pre-LayerNorm(1152)，eps=1e-6，两段残差 |
| DeepStack | deepstack_visual_indexes=[]，无该注入分支 |

单个视觉块为 `x += MHA(LayerNorm(x))`，然后 `x += MLP(LayerNorm(x))`。注意力按输入的空间片段处理，图中“非因果”不表示所有视频帧必然进行一次全局时空注意力。

Merger 先对 1152 维做 LayerNorm，再把每个 2×2 空间组拼为 4608 维：`[N,1152] → [N/4,4608] → Linear(4608,4608) → GELU → Linear(4608,5120)`。两个线性层都有 bias，Merger 全部 BF16。最终视觉 token 数为 `grid_t * grid_h * grid_w / 4`，N 指合并前 patch 数，不能把 2304 个位置嵌入误读为固定视觉 token 数。

## 6. MTP 辅助分支

检查点中确实存在 `mtp.fc`、`mtp.pre_fc_norm_embedding`、`mtp.pre_fc_norm_hidden`、`mtp.layers.0.*` 和 `mtp.norm`，不是仅有配置字段。

逻辑结构：主干隐藏状态与后移一个 token 的 embedding 分别做 RMSNorm → 按 `[embedding, hidden]` 拼接为 10240 → `mtp.fc` 投影至 5120 → 1 个全注意力 Decoder → `mtp.norm` → 复用主干 LM Head。

`mtp.fc.weight=[5120,10240]`，BF16；MTP 的 attention/MLP 形状与主干全注意力层一致，投影权重为 INT8。`mtp_use_dedicated_embeddings=false`；不存在独立的 `mtp.embed_tokens` / `mtp.lm_head` 权重。复用 embedding/head 不表示主干 embedding 与 head 自身绑权。

MTP 是独立辅助模块，不能把主干写成 65 层。是否启用 speculative decoding、如何传递隐藏状态与位置等由运行引擎决定。现有 Transformers 通用主干实现可忽略 mtp 权重；本图根据本地 SGLang MTP 路径说明辅助模块连接关系，不声称已验证某个实际部署运行方式。

## 7. 量化与参数统计

量化 YAML：激活为 per-token INT8 symmetric minmax；权重为 per-channel INT8 symmetric minmax。保存说明为 `W8A8_DYNAMIC`；各量化权重附有 FP32 `weight_scale` / `weight_offset`。`FLOAT` 是量化类别而不是 FP32 同义词：大多数 FLOAT 权重实际为 BF16，视觉量化投影的 bias 为 FP32。

已对全部层验证：

- 文本主干/MTP 的 MLP 与全注意力 Q/K/V/O 投影：W8A8_DYNAMIC。
- GDN 的 in_proj_qkv / in_proj_z：W8A8_DYNAMIC；out_proj、in_proj_a/b、conv1d、norm、A_log、dt_bias：FLOAT / BF16。
- VisionBlock 的 qkv、proj、linear_fc1 权重：W8A8_DYNAMIC；linear_fc2：BF16。
- Patch embedding、视觉位置嵌入、Merger、文本 embedding / LM head、各 norm 权重：BF16。

| 部分 | 参数元素数（不含量化 scale/offset） |
|---|---:|
| 64 层 Decoder | {counts['decoder_layers']:,} |
| Token Embedding | {counts['token_embedding']:,} |
| LM Head | {counts['lm_head']:,} |
| 最终 RMSNorm | {counts['final_norm']:,} |
| Vision Encoder + Merger | {counts['vision']:,} |
| 主干小计（含视觉） | {MAIN:,} |
| MTP | {counts['mtp']:,} |
| 全部模型参数 | {TOTAL:,} |

INT8 参数 {dtypes['I8']:,} 个（{dtypes['I8']/TOTAL*100:.2f}%）；BF16 参数 {dtypes['BF16']:,} 个；FP32 参数 {dtypes['F32']:,} 个。上述计数包含 bias、norm 与状态系数，不含量化 scale/offset。

所有存储张量（含量化元数据）有效数据共 {logical_bytes:,} 字节，即 {logical_bytes/1e9:.6f} GB / {logical_bytes/1024**3:.6f} GiB，与索引 metadata.total_size 一致。这个数字不是模型实际推理显存需求，不包含激活、KV/递推缓存、运行时工作区或文件头。

## 8. 复现与完整证据

```bash
python3 /home/zhujianwei/Qwen3.8-27B_architecture_assets/generate_architecture.py
```

需要 Pillow，以及脚本 FONT 指向的中文 TrueType 字体。脚本重新读取源目录的元数据、验证全部层的关键权重维度与 dtype，再输出图文件；不导入 torch、也不加载张量数据。SVG 内嵌原始中文字体，PNG/PDF 为同一场景的栅格输出。

- `Qwen3.8-27B_architecture_assets/verified_metadata.json`：配置、精确计数、输入文件 SHA-256、分片 JSON 头 SHA-256。
- `Qwen3.8-27B_architecture_assets/tensor_inventory.tsv`：全部 {len(meta)} 个张量的名称、形状、dtype、量化类型及分片。
- `Qwen3.8-27B_architecture_assets/A_overview.png` 至 `G_quantization.png`：七个放大分图。

本图以静态结构分析为目的。没有执行前向测试，因此不对实际后端兼容性、吞吐或数值精度作测试结论。
'''
(ROOT/(STEM+'_notes.md')).write_text(notes)
print('Outputs:')
for ext in ['svg','png','pdf']:
    p=ROOT/(STEM+'.'+ext);print(p,p.stat().st_size)
print(ROOT/(STEM+'_notes.md'))
print('Small labels:',len(s.small),'(only intentional badges expected)')
