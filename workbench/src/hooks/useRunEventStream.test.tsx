import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { useRunEventStream } from "./useRunEventStream";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  private readonly listeners = new Map<
    string,
    Array<(event: Event) => void>
  >();

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject
  ) {
    const callback =
      typeof listener === "function"
        ? listener
        : (event: Event) => listener.handleEvent(event);
    const current = this.listeners.get(type) ?? [];
    current.push(callback);
    this.listeners.set(type, current);
  }

  close(code = 1000, reason = "") {
    this.emit(closeEvent(code, reason));
  }

  emit(event: Event) {
    for (const listener of this.listeners.get(event.type) ?? []) {
      listener(event);
    }
  }
}

function closeEvent(code: number, reason: string): Event {
  return Object.assign(new Event("close"), { code, reason });
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  vi.spyOn(api, "runEventsSocketUrl").mockReturnValue(
    "ws://127.0.0.1:5173/api/lab/runs/run-1/events/ws"
  );
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useRunEventStream", () => {
  it("deduplicates replayed event sequences", () => {
    const onEvent = vi.fn();
    const { result, unmount } = renderHook(() =>
      useRunEventStream("run-1", true, onEvent)
    );
    const socket = FakeWebSocket.instances[0];

    act(() => socket.emit(new Event("open")));
    const envelope = {
      sequence: 1,
      created_at: "2026-07-26T10:00:00+00:00",
      payload: { event: "run_claimed", attempt: 1 }
    };
    act(() => {
      socket.emit(
        new MessageEvent("message", { data: JSON.stringify(envelope) })
      );
      socket.emit(
        new MessageEvent("message", { data: JSON.stringify(envelope) })
      );
    });

    expect(result.current.connection).toBe("live");
    expect(result.current.events).toEqual([envelope]);
    expect(onEvent).toHaveBeenCalledTimes(1);
    unmount();
  });

  it("reconnects after an abnormal close", () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(() =>
      useRunEventStream("run-1", true)
    );
    const first = FakeWebSocket.instances[0];

    act(() => first.emit(closeEvent(1006, "worker connection lost")));
    expect(result.current.connection).toBe("reconnecting");
    expect(result.current.error).toContain("worker connection lost");

    act(() => vi.advanceTimersByTime(1_000));
    expect(FakeWebSocket.instances).toHaveLength(2);
    unmount();
  });
});
