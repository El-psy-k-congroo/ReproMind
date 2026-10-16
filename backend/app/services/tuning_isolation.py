"""Give each search trial its own writable source, data and checkpoint paths."""
import os
import shutil
from pathlib import Path
from .source_selection import native_path


def isolate_trial(argv,cwd,env,target):
    source=native_path(Path(env.get('PYTHONPATH') or cwd).resolve())
    cwd=native_path(Path(cwd).resolve());target=native_path(target)
    if cwd!=source and source not in cwd.parents: raise ValueError('调参 cwd 不属于当前源码根目录')
    if source==target or source in target.parents: raise ValueError('调参隔离目录不能位于原源码内部')
    excluded={'.git','.venv','.conda','__pycache__','node_modules'}
    files=[];total=0
    for directory,dirs,names in os.walk(source,followlinks=False):
        parent=Path(directory)
        if any((parent/name).is_symlink() for name in [*dirs,*names]): raise ValueError('调参源码包含链接，无法保证检查点隔离')
        dirs[:]=[name for name in dirs if name not in excluded]
        for name in names:
            path=parent/name;total+=path.stat().st_size;files.append(path)
            if len(files)>20000 or total>4_000_000_000: raise ValueError('调参源码和数据超过隔离复制预算；需先缩小运行输入，未启动训练')
    target.parent.mkdir(parents=True,exist_ok=True)
    if total>shutil.disk_usage(target.parent).free-2_000_000_000: raise ValueError('磁盘不足以隔离调参输入，未启动训练')
    target.mkdir(exist_ok=False)
    for path in files:
        destination=target/path.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)  # Never hard-link writable author checkpoints.
    canonical=str(Path(source));ordinary=canonical[4:] if canonical.startswith('\\\\?\\') else canonical
    def remap(value):
        return str(value).replace(canonical,str(target)).replace(ordinary,str(target))
    mapped=[remap(value) for value in argv];mapped_env={key:remap(value) for key,value in env.items()}
    mapped_env['PYTHONPATH']=str(target);mapped_env['REPROMIND_OUTPUT_DIR']=str(target.parent/'output')
    for index,value in enumerate(mapped[:-1]):
        if value in {'-o','--output','--output-dir','--output_dir','--work-dir','--work_dir','--save-dir','--save_dir'}:
            mapped[index+1]=str(target.parent/'output')
    return mapped,target/cwd.relative_to(source),mapped_env,{'source':str(source),'copy':str(target),'file_count':len(files),'bytes':total}
