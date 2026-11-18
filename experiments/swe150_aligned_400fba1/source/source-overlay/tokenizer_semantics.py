"""Pinned Qwen tokenizer structural contract plus bounded semantic qualification.

Never claims equivalence for every string or native/process security.
"""
import hashlib
import json
from pathlib import Path

PINS={'tokenizer-config.json':'5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b','tokenizer.json':'5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42','model-config.json':'69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c'}
OBSERVED={'pre_tokenizer':'d730489d91319a99960af82e8fed44be8533ca92615198807b41c29dda5556fc','decoder':'1d64d97add535d9ad91561aabea254849cf7f2ea4b924cc61c17152f1dd6e672'}
FLAGS=('single_word','lstrip','rstrip','normalized','special')
BASE_CORPUS=('', ' ', '  ', '\t','\n','\r\n','a\r\nb\tc', ' a ', 'a  b',
 "I'm we're they've can't I'd he'll IT'S",'...!?--()[]{}<>/@#$% 1234567890',
 'café','cafe\u0301','\u0301','a\u0301\u0327','中\u0301文','中文 Ελληνικά 😀👩\u200d💻',
 '<tool_call>{"name":"bash","arguments":{"cmd":"pytest"}}</tool_call>',
 '<tool_response>AssertionError: expected 2, got 1\n</tool_response>',
 '<|im_start|>user\nInspect the test.<|im_end|>\n', 'def add(a,b):\n\treturn a+b\n')

class SemanticError(RuntimeError):pass
def need(ok,message):
    if not ok:raise SemanticError(message)
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def pinned_json(path,expected):
    raw=Path(path).read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'pinned artifact hash differs: '+Path(path).name)
    return json.loads(raw)

def normalize_merges(rows):
    need(type(rows) is list,'BPE merges must be an ordered list')
    result=[]
    for row in rows:
        if type(row) is str:
            # Literal ASCII separator only. No whitespace stripping or guessing.
            pair=row.split(' ')
        else:pair=row
        need(type(pair) in (list,tuple) and len(pair)==2 and all(type(x) is str and x and ' ' not in x for x in pair),'ambiguous BPE merge pair')
        result.append(tuple(pair))
    return result

def expected_added(config):
    values=config['added_tokens_decoder'];need(type(values) is dict,'added decoder must be a mapping')
    result={}
    for key,row in values.items():
        need(str(int(key))==key and int(key)>=0 and type(row) is dict and set(row)=={'content',*FLAGS},'invalid configured added-token record')
        need(type(row['content']) is str and row['content'] and all(type(row[f]) is bool for f in FLAGS),'invalid added-token content/flags')
        result[int(key)]=dict(row,id=int(key))
    need(len({r['content'] for r in result.values()})==len(result),'duplicate configured token content')
    return result

def validate_structure(raw,loaded,config):
    need(set(raw)==set(loaded),'tokenizer top-level fields differ')
    for key in set(raw)-{'model','added_tokens','pre_tokenizer','decoder'}:need(canonical(raw[key])==canonical(loaded[key]),'strict tokenizer component differs: '+key)
    need(set(raw['model'])==set(loaded['model']),'model fields differ')
    for key in set(raw['model'])-{'merges'}:need(canonical(raw['model'][key])==canonical(loaded['model'][key]),'model/vocab/IDs differ: '+key)
    need(raw['model']['type']=='BPE','only pinned BPE supported')
    merges=normalize_merges(raw['model']['merges'])
    need(merges==normalize_merges(loaded['model']['merges']),'BPE merge order/content differs')
    expected=expected_added(config)
    def index(rows):
        need(type(rows) is list and all(type(r) is dict and set(r)=={'id','content',*FLAGS} and type(r['id']) is int for r in rows),'invalid loaded added-token record')
        result={r['id']:r for r in rows};need(len(result)==len(rows),'duplicate added-token ID');return result
    actual=index(loaded['added_tokens']);original=index(raw['added_tokens'])
    need(actual==expected,'loaded added tokens differ from pinned config')
    need(all(expected.get(k)==row for k,row in original.items()),'raw tokens conflict with pinned config')
    for key in OBSERVED:
        need(loaded[key]==raw[key] or digest(loaded[key])==OBSERVED[key],'unobserved '+key+' structure/regex differs')
    return {'base_vocab_ids':'EXACT','ordered_merges':'NORMALIZED_EXACT','model_other_fields':'EXACT','normalizer':'EXACT','post_processor':'EXACT','added_tokens':'CONFIG_EXACT','configured_added_count':len(expected),'config_derived_ids':sorted(expected.keys()-original.keys()),'structure_differences':{k:{'raw_sha256':digest(raw[k]),'loaded_sha256':digest(loaded[k])} for k in sorted(raw) if raw[k]!=loaded[k]}}

def special_mapping(config,tokenizer):
    named={key:config.get(key) for key in ('bos_token','eos_token','unk_token','sep_token','pad_token','cls_token','mask_token') if config.get(key) is not None}
    extra=config.get('extra_special_tokens',config.get('additional_special_tokens',[]))
    # Exact precedence in the pinned Transformers v5 source: a present named
    # extra_special_tokens dict takes precedence over legacy additional list.
    if isinstance(extra,dict):named.update(extra);extra_list=[]
    else:extra_list=extra
    need(dict(tokenizer.special_tokens_map)==named,'special-token named mapping differs')
    need(list(map(str,tokenizer.extra_special_tokens))==extra_list,'extra special-token mapping differs')
    expected=expected_added(config);ids={row['content']:i for i,row in expected.items()}
    for key,content in named.items():
        need(getattr(tokenizer,key)==content and getattr(tokenizer,key+'_id')==ids[content],'special-token attribute/ID differs: '+key)
    for key in ('bos_token','eos_token','unk_token','pad_token'):
        if config.get(key) is None:need(getattr(tokenizer,key) is None and getattr(tokenizer,key+'_id') is None,'unexpected null special mapping: '+key)
    tokens=list(dict.fromkeys([*named.values(),*extra_list]))
    need(set(tokenizer.all_special_tokens)==set(tokens) and len(tokenizer.all_special_tokens)==len(tokens),'all-special-token set differs')
    need(list(tokenizer.all_special_ids)==[ids[t] for t in tokenizer.all_special_tokens],'all-special ID mapping differs')
    for row in expected.values():need(tokenizer.convert_tokens_to_ids(row['content'])==row['id'],'configured token ID differs')
    return {'named':named,'named_ids':{k:ids[v] for k,v in named.items()},'legacy_additional_ids':{v:ids[v] for v in config.get('additional_special_tokens',[])},'all_special_ids':list(tokenizer.all_special_ids)}

def compare_encoding(tokenizer,reference,raw_reference,text,new_contents):
    backend=tokenizer.backend_tokenizer;result=[]
    for special in (False,True):
        actual=backend.encode(text,add_special_tokens=special).ids
        hf=tokenizer.encode(text,add_special_tokens=special)
        expected=reference.encode(text,add_special_tokens=special).ids
        raw=raw_reference.encode(text,add_special_tokens=special).ids
        need(actual==hf==expected,'encoding/boundary mismatch for text SHA '+hashlib.sha256(text.encode()).hexdigest())
        contains_new=any(t in text for t in new_contents)
        need(contains_new or raw==actual,'raw tokenizer semantic mismatch outside config-derived tokens')
        for skip in (False,True):
            decoded=backend.decode(actual,skip_special_tokens=skip)
            need(decoded==reference.decode(expected,skip_special_tokens=skip),'decoder semantics differ')
            need(tokenizer.decode(actual,skip_special_tokens=skip,clean_up_tokenization_spaces=False)==decoded,'HF decode differs')
            if raw==actual:need(raw_reference.decode(raw,skip_special_tokens=skip)==decoded,'raw decoder semantics differ')
            # Exact decode/encode roundtrip relative to the configured reference,
            # not false reconstruction of pre-NFC input or stripped specials.
            need(backend.encode(decoded,add_special_tokens=False).ids==reference.encode(decoded,add_special_tokens=False).ids,'roundtrip semantics differ')
        result.append({'special_insertion':special,'ids_sha256':digest(actual),'token_count':len(actual),'raw_equal':raw==actual,'raw_ids_sha256':digest(raw),'config_derived_text':contains_new})
    return {'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'comparisons':result}

class TokenizerSemanticGuard:
    def __init__(self,evidence_dir,tokenizer,*,model_layout=False):
        self.root=Path(evidence_dir);self.tokenizer=tokenizer
        names={'tokenizer-config.json':'tokenizer_config.json','model-config.json':'config.json'} if model_layout else {}
        self.paths={key:self.root/names.get(key,key) for key in PINS}
        values={key:pinned_json(self.paths[key],sha) for key,sha in PINS.items()}
        self.raw=values['tokenizer.json'];self.config=values['tokenizer-config.json']
        self.loaded=json.loads(tokenizer.backend_tokenizer.to_str())
        self.structure=validate_structure(self.raw,self.loaded,self.config)
        self.mapping=special_mapping(self.config,tokenizer)
        from tokenizers import Tokenizer,AddedToken
        self.raw_reference=Tokenizer.from_file(str(self.paths['tokenizer.json']))
        self.reference=Tokenizer.from_file(str(self.paths['tokenizer.json']))
        configured=expected_added(self.config)
        for i,row in sorted(configured.items()):
            token=AddedToken(row['content'],**{k:row[k] for k in FLAGS})
            if row['special']:self.reference.add_special_tokens([token])
            else:self.reference.add_tokens([token])
            need(self.reference.token_to_id(row['content'])==i,'configured reference assigned different token ID')
        need({r['id']:r for r in json.loads(self.reference.to_str())['added_tokens']}==configured,'configured reference flags differ')
        self.new_contents=[configured[i]['content'] for i in self.structure['config_derived_ids']]
        self.backend_snapshot=digest(self.loaded)
        texts=list(BASE_CORPUS)
        for i,row in sorted(configured.items()):
            content=row['content'];need(tokenizer.encode(content,add_special_tokens=False)==[i],'configured token is not a single exact ID')
        for content in self.new_contents:
            texts.extend([content,'x'+content+'y',' '+content+' ', '\t'+content+'\r\n',content+content,'a\u0301'+content+'中',content[:-1],content[1:]])
        self.cases=[self.check_text(text) for text in texts]
        self.backend=tokenizer.backend_tokenizer
        self.verify(full=True)
    def check_text(self,text):return compare_encoding(self.tokenizer,self.reference,self.raw_reference,text,self.new_contents)
    def verify(self,full=False):
        # The pinned files and the full backend digest are checked once, at startup (full=True). Per request
        # (every renderer identity check) only cheap drift signals remain: re-reading 13 MB of pinned JSON and
        # digesting the 8 MB backend took ~0.7 s per call, several calls per request (v10/v11 measurements),
        # and even len(tokenizer) rebuilds the 248K-entry vocabulary (~45 ms).
        if full:
            for key,sha in PINS.items():pinned_json(self.paths[key],sha)
            need(digest(json.loads(self.tokenizer.backend_tokenizer.to_str()))==self.backend_snapshot,'validated loaded backend changed')
        need(self.tokenizer.backend_tokenizer is self.backend,'validated loaded backend changed')
        need(special_mapping(self.config,self.tokenizer)==self.mapping,'validated special mapping changed')
        return True
    def receipt(self):return {'status':'PINNED_TOKENIZER_SEMANTIC_CORPUS_PASS_NOT_SYSTEM_VALID','structure':self.structure,'special_mapping':self.mapping,'corpus_count':len(self.cases),'corpus_receipt_sha256':digest(self.cases),'corpus_sample':self.cases[:6],'config_derived_token_boundary_cases':len(self.new_contents)*8,'universal_equivalence_claimed':False,'native_closure_complete':False,'system_valid':False}
