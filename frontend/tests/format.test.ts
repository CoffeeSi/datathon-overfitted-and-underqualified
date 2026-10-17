import { describe, expect, it } from "vitest";
import { timecode } from "../lib/format";

describe("displayed timecodes", () => {
  it.each([
    [28.4, "00:28.4"],
    [59.999, "00:59.9"],
    [0.09, "00:00.0"],
    [0.3, "00:00.3"],
    [60, "01:00.0"],
    [3600.4, "01:00:00.4"],
  ] as const)("formats %s seconds without floating-point drift or rounding into the next second", (seconds, expected) => {
    expect(timecode(seconds, true)).toBe(expected);
  });

  it.each([Number.NaN, Infinity, -Infinity, -1])("does not fabricate a timestamp for invalid value %s", (seconds) => {
    expect(timecode(seconds, true)).toBe("—:—");
  });
});
