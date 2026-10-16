from app.services.reproduction_engine import _extract_metrics


def test_final_checkpoint_accuracy_does_not_inherit_last_epoch_training_loss():
    output = '''>>> Epoch 32
[train] loss = 0.278
[val] acc = 0.85
[test] acc = 0.84
Loading 16th epoch
Final evaluation
[test] acc = 0.861
'''
    assert _extract_metrics(output) == {'accuracy': .861}


def test_training_or_validation_metrics_cannot_satisfy_final_test_acceptance():
    assert _extract_metrics('Epoch 1 train acc:99 loss:0.1 | val acc:85 loss:0.2') == {}


def test_test_summary_marked_after_metrics_and_later_train_values_remain_separate():
    output = ('* Epoch: [0/2] Top 1-err 95 Top 5-err 84 Train Loss 4.3\n'
              '* Epoch: [0/2] Top 1-err 91 Top 5-err 70 Test Loss 3.9\n'
              '* Epoch: [1/2] Top 1-err 88 Top 5-err 66 Train Loss 3.8\n'
              '* Epoch: [1/2] Top 1-err 84 Top 5-err 58 Test Loss 3.5\n'
              '[train] loss:0.1\n')
    assert _extract_metrics(output) == {'top1error': .84, 'top5error': .58, 'loss': 3.5}


def test_unlabelled_legacy_outputs_keep_values_without_fabricating_a_test_label():
    assert _extract_metrics('Average accuracy: tensor([0.82])') == {'accuracy': .82}
