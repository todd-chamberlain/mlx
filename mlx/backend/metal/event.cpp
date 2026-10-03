// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// EventImpl implementations
///////////////////////////////////////////////////////////////////////////////

namespace metal {

EventImpl::EventImpl(Device& d) {
  auto p = new_scoped_memory_pool();
  mtl_event_ = NS::TransferPtr(d.mtl_device()->newSharedEvent());
  if (!mtl_event_) {
    throw std::runtime_error(
        "[Event::Event] Failed to create Metal shared event.");
  }
}

EventImpl::~EventImpl() {
  auto p = new_scoped_memory_pool();
  mtl_event_.reset();
}

void EventImpl::wait(uint64_t value) {
  mtl_event_->waitUntilSignaledValue(value, -1); // never times out
}

void EventImpl::signal(uint64_t value) {
  mtl_event_->setSignaledValue(value);
}

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::wait() {
  check_error();
  cast<metal::EventImpl>().wait(value());
  check_error();
}

namespace {

// Flush every GPU stream's pending command buffer. Used before blocking the
// host on a CPU-signaled event: the CPU work may itself depend on GPU work
// that is still sitting unencoded/uncommitted in these buffers (see #4556).
void commit_all_gpu_streams() {
  auto pool = metal::new_scoped_memory_pool();
  for (auto& [index, encoder] : metal::get_command_encoders()) {
    encoder.end_encoding();
    encoder.commit();
  }
}

} // namespace

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::wait_event(stream, *this, [value = value()](Event& self) {
      self.cast<metal::EventImpl>().wait(value);
    });
  } else if (this->stream().device == Device::cpu) {
    // The event will be signaled from a CPU stream (e.g. a distributed
    // collective, or any slow CPU op). Encoding a hardware wait would leave a
    // *committed* command buffer blocked on host work of unbounded duration,
    // and Metal's command-buffer watchdog (~5s) then fails it with
    // kIOGPUCommandBufferCallbackErrorTimeout. The GPU has nothing to do
    // until the host signals anyway, so perform the wait on the host instead:
    // flush pending GPU work first (the CPU work may depend on it), then block.
    commit_all_gpu_streams();
    wait();
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.wait_event(*this, value());
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::signal_event(stream, *this, [value = value()](Event& self) {
      self.cast<metal::EventImpl>().signal(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.signal_event(*this, value());
  }
}

bool Event::is_signaled() const {
  auto* mtl_event = cast<metal::EventImpl>().mtl_event();
  return mtl_event->signaledValue() >= value();
}

std::atomic<Error*>& Event::error() {
  return cast<metal::EventImpl>().error();
}

} // namespace mlx::core
