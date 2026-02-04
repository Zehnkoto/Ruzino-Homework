#pragma once

#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>

#include <functional>
#include <mutex>

#include "api.h"
#include "stage_listener/api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

class STAGE_LISTENER_API StageListener : public pxr::TfWeakBase {
   public:
    using DirtyPathSet = std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash>;

    // 回调函数类型
    using PrimAddedCallback = std::function<void(const pxr::UsdPrim&)>;
    using PrimRemovedCallback = std::function<void(const pxr::SdfPath&)>;
    using PrimChangedCallback = std::function<void(const pxr::SdfPath&)>;

    explicit StageListener(const pxr::UsdStagePtr& stage);

    void CapturePrimSnapshot();

    std::mutex& GetMutex();

    void GetDirtyPaths(DirtyPathSet& outDirtyPaths);

    // 注册回调函数
    void SetPrimAddedCallback(PrimAddedCallback callback)
    {
        prim_added_callback_ = std::move(callback);
    }

    void SetPrimRemovedCallback(PrimRemovedCallback callback)
    {
        prim_removed_callback_ = std::move(callback);
    }

    void SetPrimChangedCallback(PrimChangedCallback callback)
    {
        prim_changed_callback_ = std::move(callback);
    }

   private:
    // 处理Prim结构变化（添加/删除）
    void OnStageContentsChanged(
        const pxr::UsdNotice::StageContentsChanged& notice);

    // 处理属性变化（如变换、几何数据）
    void OnObjectsChanged(const pxr::UsdNotice::ObjectsChanged& notice);

    pxr::UsdStagePtr stage_;
    std::mutex mutex_;
    DirtyPathSet dirtyPaths_;  // 脏Prim路径集合
    pxr::TfNotice::Key stageContentsChangedKey_, objectsChangedKey_;
    DirtyPathSet previousPrimPaths_;  // 上一帧的Prim路径集合

    // 回调函数
    PrimAddedCallback prim_added_callback_;
    PrimRemovedCallback prim_removed_callback_;
    PrimChangedCallback prim_changed_callback_;
};

RUZINO_NAMESPACE_CLOSE_SCOPE