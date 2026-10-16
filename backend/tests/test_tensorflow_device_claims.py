from app.api.experiment_branches import _gpu_preview_issues

def test_session_run_device_context_is_not_graph_placement(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nwith tf.device("/gpu:0"):\n    sess.run(train_op)\n')
    assert any('Session.run' in issue for issue in _gpu_preview_issues(tmp_path, {}))

def test_fabricated_training_device_log_is_rejected(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nprint("Training ops device: /gpu:0")\n')
    assert any('固定字符串' in issue for issue in _gpu_preview_issues(tmp_path, {}))

def test_graph_placement_and_runtime_metadata_are_not_rejected(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nwith tf.device("/gpu:0"):\n    a = tf.constant(1)\nmetadata = tf.RunMetadata()\n')
    assert not _gpu_preview_issues(tmp_path, {})

def test_device_profile_cannot_add_optimizer_step_outside_loop(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nmetadata = tf.RunMetadata()\nsess.run(model.opt_op, run_metadata=metadata)\n')
    assert any('额外执行' in issue for issue in _gpu_preview_issues(tmp_path, {}))

def test_profile_on_original_training_loop_is_allowed(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nmetadata = tf.RunMetadata()\nfor epoch in range(58):\n    sess.run(model.opt_op, run_metadata=metadata)\n')
    assert not _gpu_preview_issues(tmp_path, {})

def test_duplicate_optimizer_inside_first_epoch_is_rejected(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nmetadata = tf.RunMetadata()\nfor epoch in range(58):\n    sess.run(model.opt_op)\n    if epoch == 0:\n        sess.run(model.opt_op, run_metadata=metadata)\n')
    assert any('重复执行' in issue for issue in _gpu_preview_issues(tmp_path, {}))

def test_mutually_exclusive_traced_step_does_not_duplicate_optimizer(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nmetadata = tf.RunMetadata()\nfor epoch in range(58):\n    if epoch == 0:\n        sess.run(model.opt_op, run_metadata=metadata)\n    else:\n        sess.run(model.opt_op)\n')
    assert not _gpu_preview_issues(tmp_path, {})


def test_full_trace_each_epoch_is_rejected_but_conditional_options_are_allowed(tmp_path):
    path = tmp_path / 'train.py'
    header = 'import tensorflow as tf\nmetadata = tf.RunMetadata()\nfor epoch in range(58):\n'
    path.write_text(header + '    sess.run(model.opt_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))\n')
    assert any('每轮' in issue for issue in _gpu_preview_issues(tmp_path, {}))
    path.write_text(header + '    sess.run(model.opt_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if epoch == 0 else None)\n')
    assert not _gpu_preview_issues(tmp_path, {})
