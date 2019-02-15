package simulator;

import java.util.Objects;

// StateTuple has two boolean value fromSlip and fromBreakdown to indicate whether the State
//   are recovered from Slip or Breakdown. Since the simulator has already handled the slip or
//   breakdown for us (the returned state's slip and breakdown are always false, the simulator
//   only returns recovered states). However, we need to know whether the returned state are
//   recovered from slip or breakdown, otherwise we cannot tell whether the state are generated
//   from moved 0 cells or moved but slipped or moved but breakdown.
public class StateTuple {
    public final boolean fromSlip;
    public final boolean fromBreakdown;
    public final State state;

    public StateTuple(State state, boolean fromSlip, boolean fromBreakdown) {
        this.state = state;
        this.fromSlip = fromSlip;
        this.fromBreakdown = fromBreakdown;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StateTuple that = (StateTuple) o;
        return fromSlip == that.fromSlip &&
                fromBreakdown == that.fromBreakdown &&
                Objects.equals(state, that.state);
    }

    @Override
    public int hashCode() {
        return Objects.hash(state, fromSlip, fromBreakdown);
    }

    @Override
    public String toString() {
        return "StateTuple{" +
                "fromSlip=" + fromSlip +
                ", fromBreakdown=" + fromBreakdown +
                ", state=" + state +
                '}';
    }

}