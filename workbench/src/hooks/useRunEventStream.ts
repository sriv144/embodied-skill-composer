import { useEffect, useRef, useState } from "react";
import { api, formatApiError } from "../api";
import type { RunEventEnvelope } from "../types";

export type RunEventConnection =
  | "idle"
  | "connecting"
  | "live"
  | "reconnecting"
  | "closed"
  | "error";

export type RunEventStream = {
  connection: RunEventConnection;
  events: RunEventEnvelope[];
  error: string | null;
};

function isRunEventEnvelope(value: unknown): value is RunEventEnvelope {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<RunEventEnvelope>;
  return (
    typeof candidate.sequence === "number" &&
    typeof candidate.created_at === "string" &&
    typeof candidate.payload === "object" &&
    candidate.payload !== null &&
    "event" in candidate.payload &&
    typeof candidate.payload.event === "string"
  );
}

export function useRunEventStream(
  runId: string | null,
  enabled: boolean,
  onEvent?: (event: RunEventEnvelope) => void
): RunEventStream {
  const [connection, setConnection] = useState<RunEventConnection>("idle");
  const [events, setEvents] = useState<RunEventEnvelope[]>([]);
  const [error, setError] = useState<string | null>(null);
  const onEventRef = useRef(onEvent);

  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    const socketUrl = runId && enabled ? api.runEventsSocketUrl(runId) : null;
    if (!runId || !enabled || !socketUrl) {
      setConnection("idle");
      setEvents([]);
      setError(null);
      return;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let reconnectAttempt = 0;
    let lastSequence = 0;

    const connect = () => {
      if (disposed) return;
      setConnection(reconnectAttempt === 0 ? "connecting" : "reconnecting");
      socket = new WebSocket(socketUrl);

      socket.addEventListener("open", () => {
        reconnectAttempt = 0;
        setConnection("live");
        setError(null);
      });

      socket.addEventListener("message", (message) => {
        try {
          const envelope: unknown = JSON.parse(String(message.data));
          if (!isRunEventEnvelope(envelope)) {
            throw new Error("The run event payload did not match the typed event contract.");
          }
          if (envelope.sequence <= lastSequence) return;
          lastSequence = envelope.sequence;
          setEvents((current) => [...current, envelope].slice(-100));
          onEventRef.current?.(envelope);
        } catch (reason) {
          setConnection("error");
          setError(formatApiError(reason));
        }
      });

      socket.addEventListener("close", (event) => {
        if (disposed) return;
        if (event.code === 1000) {
          setConnection("closed");
          return;
        }
        reconnectAttempt += 1;
        const delay = Math.min(8_000, 500 * 2 ** Math.min(reconnectAttempt, 4));
        setConnection("reconnecting");
        setError(
          event.reason ||
            `Live run updates disconnected (code ${event.code}); retrying.`
        );
        reconnectTimer = window.setTimeout(connect, delay);
      });

      socket.addEventListener("error", () => {
        setConnection("error");
        setError("The live run event connection failed.");
      });
    };

    setEvents([]);
    setError(null);
    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close(1000, "view changed");
    };
  }, [enabled, runId]);

  return { connection, events, error };
}
