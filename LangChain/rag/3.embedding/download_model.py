from modelscope import snapshot_download

snapshot_download(
    model_id="BAAI/bge-base-zh-v1.5",
    local_dir="~/models/bge-base-zh-v1.5"
)
print("🎉 下载完成")