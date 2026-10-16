from app.services.source_calculations import calculation_packets, numbered_code


def test_metrics_are_retrieved_by_expression_not_function_name():
    body = '''def evaluate_dataset(original, restored):
    diff = original - restored
    prd = torch.sqrt(torch.sum(diff**2)/torch.sum(original**2))*100
    rmse = torch.sqrt(torch.sum(diff**2)/torch.numel(diff))
    return prd, rmse
'''
    packets = calculation_packets(body, '解释 PRD、RMSE 的实际计算方法', 'evaluate.py')
    joined='\n'.join(row['code'] for row in packets)
    assert 'diff = original - restored' in joined and 'torch.numel(diff)' in joined
    assert 'scope' in packets[0] and packets[0]['file'] == 'evaluate.py'


def test_preprocessing_and_printed_ratio_are_bound_to_question():
    body = '''def compress(chunk):
    chunk = np.cbrt(chunk)
    code = model.encode(chunk)
    print(f"CR: {original_bytes/compressed_bytes}")
    return code
'''
    packets = calculation_packets(body, '解释立方根和压缩率', 'compress.py')
    joined='\n'.join(row['code'] for row in packets)
    assert 'np.cbrt(chunk)' in joined and 'original_bytes/compressed_bytes' in joined
    assert not calculation_packets(body, '说明 optimizer CUDA 设置', 'compress.py')


def test_packet_does_not_cut_formula_mid_line_or_keep_comment_banners():
    lines=['# long comment '*80, 'result = original - restored', 'ratio = numerator / denominator']
    packet=numbered_code(lines,0,len(lines),limit=50)
    assert packet=='L2: result = original - restored'
