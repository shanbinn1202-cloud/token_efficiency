"""Reject configurations whose semantics the port cannot preserve."""


def check(config):
    b, g = config['backend'], config['generation']
    if b['name'] != 'vllm' or b['tensor_parallel_size'] != 1:
        raise ValueError('RepairKV currently supports vLLM TP=1')
    if config['model']['dtype'] != 'bfloat16' or g['temperature'] != 0:
        raise ValueError('RepairKV currently requires BF16 greedy decoding')
    if any(g[k] != v for k, v in (('top_p', 1.), ('top_k', -1), ('repetition_penalty', 1.))):
        raise ValueError('RepairKV requires top_p=1, top_k=-1, repetition_penalty=1')
    options = config['methods']['repairkv']['options']
    expected = {'repair_tokens', 'sink_size', 'recency_window', 'pooling', 'burst_left', 'burst_right'}
    if options.keys() != expected:
        raise ValueError('Unexpected RepairKV policy options')
    if any(type(value) is not int or value < 0 for key, value in options.items() if key != 'pooling'):
        raise ValueError('RepairKV integer options must be nonnegative')
    if options['pooling'] not in {'max', 'mean'}:
        raise ValueError('RepairKV pooling must be max or mean')
    if not 0 < options['repair_tokens'] < config['cache']['kv_budget']:
        raise ValueError('RepairKV requires 0 < repair_tokens < kv_budget')
