import { describe, expect, it } from "vitest";
import { actionName, reviewEvents } from "../features/review/presentation";

describe("untrusted finding types", () => {
  it.each(["constructor", "toString", "__proto__"])("keeps prototype name %s as an unknown type", (type) => {
    expect(actionName(type)).toBe("Событие для проверки");
  });
  it("preserves distinct events at the same timestamp", () => {
    const finding = { type:"constructor", timecode_sec: 2, description:"Synthetic", evidence_frames:[] };
    const events = reviewEvents([finding, finding]);
    expect(events[0].id).not.toBe(events[1].id);
    expect(events.map(event => event.originalIndex)).toEqual([0,1]);
  });
});
