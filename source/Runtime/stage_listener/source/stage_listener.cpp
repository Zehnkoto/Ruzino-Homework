#include "stage_listener/stage_listener.h"

RUZINO_NAMESPACE_OPEN_SCOPE

StageListener::StageListener(const pxr::UsdStagePtr& stage) : stage_(stage)
{
    // 检查stage是否有效
    if (!stage_) {
        throw std::runtime_error("Invalid stage pointer");
    }

    // 注册监听场景内容变化（Prim添加/删除）
    stageContentsChangedKey_ = pxr::TfNotice::Register(
        pxr::TfCreateWeakPtr(this),
        &StageListener::OnStageContentsChanged,
        stage_);

    // 注册监听属性变化
    objectsChangedKey_ = pxr::TfNotice::Register(
        pxr::TfCreateWeakPtr(this), &StageListener::OnObjectsChanged, stage_);
}

void StageListener::CapturePrimSnapshot()
{
    std::lock_guard<std::mutex> lock(mutex_);
    previousPrimPaths_.clear();
    if (stage_) {
        for (const auto& prim : stage_->Traverse()) {
            previousPrimPaths_.insert(prim.GetPath());
        }
    }
}

std::mutex& StageListener::GetMutex()
{
    return mutex_;
}

void StageListener::GetDirtyPaths(DirtyPathSet& outDirtyPaths)
{
    std::lock_guard<std::mutex> lock(mutex_);
    outDirtyPaths = std::move(dirtyPaths_);
    dirtyPaths_.clear();  // 确保原容器为空
}

void StageListener::OnStageContentsChanged(
    const pxr::UsdNotice::StageContentsChanged& notice)
{
    if (!stage_)
        return;

    std::lock_guard<std::mutex> lock(mutex_);

    // 获取当前所有 Prim 路径
    DirtyPathSet currentPrimPaths;
    for (const auto& prim : stage_->Traverse()) {
        currentPrimPaths.insert(prim.GetPath());
    }

    // 对比差异：新增的 Prim
    for (const auto& path : currentPrimPaths) {
        if (!previousPrimPaths_.count(path)) {
            dirtyPaths_.insert(path);  // 新增的 Prim

            // 调用回调函数
            if (prim_added_callback_) {
                auto prim = stage_->GetPrimAtPath(path);
                if (prim && prim.IsValid()) {
                    prim_added_callback_(prim);
                }
            }
        }
    }

    // 对比差异：删除的 Prim
    for (const auto& path : previousPrimPaths_) {
        if (!currentPrimPaths.count(path)) {
            dirtyPaths_.insert(path);  // 删除的 Prim

            // 调用回调函数
            if (prim_removed_callback_) {
                prim_removed_callback_(path);
            }
        }
    }

    // 更新快照为当前状态
    previousPrimPaths_.swap(currentPrimPaths);
}

void StageListener::OnObjectsChanged(
    const pxr::UsdNotice::ObjectsChanged& notice)
{
    if (!stage_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (const pxr::SdfPath& path : notice.GetChangedInfoOnlyPaths()) {
        // 标记属性变化的Prim路径
        auto prim_path = path.GetPrimPath();
        dirtyPaths_.insert(prim_path);

        // 调用回调函数
        if (prim_changed_callback_) {
            prim_changed_callback_(prim_path);
        }
    }
}

RUZINO_NAMESPACE_CLOSE_SCOPE