// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/event.h"

namespace mlx::core::metal {

class EventImpl {
 public:
  EventImpl(Device& d);
  ~EventImpl();

  void wait(uint64_t value);
  // Wait at most ``timeout_ms`` (<= 0: forever). Returns false on timeout.
  bool wait(uint64_t value, int64_t timeout_ms);
  void signal(uint64_t value);

  auto& error() {
    return error_;
  }

  auto* mtl_event() const {
    return mtl_event_.get();
  }

 private:
  // All streams outlive events so pointers would be always valid.
  std::atomic<Error*> error_;

  NS::SharedPtr<MTL::SharedEvent> mtl_event_;
};

// Bound for host-side waits on CPU-signaled events/fences, from
// MLX_HOST_WAIT_TIMEOUT_MS. <= 0 (default) means wait forever.
int64_t host_wait_timeout_ms();

} // namespace mlx::core::metal
