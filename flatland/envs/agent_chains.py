
class MotionCheck(object):
    """ Class to find chains of agents which are "colliding" with a stopped agent.
        This is to allow close-packed chains of agents, ie a train of agents travelling
        at the same speed with no gaps between them,

        The state is a *functional* graph -- each cell an agent occupies has exactly one
        out-edge, "where I want to go next" -- held in plain dicts. It used to be a
        networkx DiGraph, but it is rebuilt from scratch on every env.step and networkx's
        generality dominated step time; the dicts below are ~8x faster and depend only on
        the standard library. Visualising a MotionCheck (which is what networkx and
        graphviz were really for) now lives in notebooks/Agent-Close-Following.ipynb.
    """
    def __init__(self):
        self.nodes = {}     # rc -> None; an insertion-ordered node set
        self.succ = {}      # rc -> rc; the single out-edge
        self.pred = {}      # rc -> {rc: None}; insertion-ordered distinct predecessors
        self.agent_at = {}  # rc -> agent index, for cells an agent is departing from
        self.colors = {}    # rc -> colour string
        self.xlabels = {}   # rc -> label; debug/visualisation only

    def _add_node(self, rc):
        if rc not in self.nodes:
            self.nodes[rc] = None
            self.pred[rc] = {}

    def addAgent(self, iAg, rc1, rc2, xlabel=None):
        """ add an agent and its motion as row,col tuples of current and next position.
            The agent's current position is given an "agent" attribute recording the agent index.
            If an agent does not want to move this round (rc1 == rc2) then a self-loop edge is created.
            xlabel is used for test cases to give a label (see graphviz)
        """

        # Agents which have not yet entered the env have position None.
        # Substitute this for the row = -1, column = agent index
        if rc1 is None:
            rc1 = (-1, iAg)

        if rc2 is None:
            rc2 = (-1, iAg)

        # rc1 before rc2, matching the node insertion order nx produced via
        # add_node(rc1) then add_edge(rc1, rc2). find_conflicts iterates in this order.
        self._add_node(rc1)
        self.agent_at[rc1] = iAg
        if xlabel:
            self.xlabels[rc1] = xlabel

        self._add_node(rc2)

        # Re-adding an edge replaces the old one, as nx's add_edge did.
        rcOld = self.succ.get(rc1)
        if rcOld is not None and rcOld != rc2:
            self.pred[rcOld].pop(rc1, None)
        self.succ[rc1] = rc2
        self.pred[rc2][rc1] = None

    def _reverse_reachable(self, svSources):
        """ Every node that reaches one of svSources by following movement edges forward.

            Replaces the old per-component subgraph + reverse() + DFS: the union of the
            reverse-reachable sets is the same set, so the weakly-connected-component
            decomposition was only ever an (expensive) way of partitioning the work.
            The sources themselves are included, as dfs_postorder_nodes included its source.
        """
        svSeen = set(svSources)
        lvStack = list(svSeen)
        while lvStack:
            v = lvStack.pop()
            for u in self.pred.get(v, ()):
                if u not in svSeen:
                    svSeen.add(u)
                    lvStack.append(u)
        return svSeen

    def find_stops(self):
        """ find all the stopped agents as a set of rc position nodes
            A stopped agent is a self-loop on a cell node.
        """
        return self.find_stops2()

    def find_stops2(self):
        """ find stopped agents: the self-loops, ie cells whose successor is themselves
        """
        return { u for u, v in self.succ.items() if u == v }

    def find_stop_preds(self, svStops=None):

        if svStops is None:
            svStops = self.find_stops2()

        # Everything that chains back from a stopped agent is blocked by it.
        return self._reverse_reachable(svStops)

    def find_swaps(self):
        """ find all the swap conflicts where two agents are trying to exchange places.
            These are the 2-cycles: u wants v's cell and v wants u's.
        """
        svSwaps = set()
        for u, v in self.succ.items():
            if u != v and self.succ.get(v) == u:
                svSwaps.add(u)
                svSwaps.add(v)
        return svSwaps

    def find_same_dest(self):
        """ find groups of agents which are trying to land on the same cell.
            ie there is a gap of one cell between them and they are both landing on it.
        """
        pass

    def find_conflicts(self):
        svStops = self.find_stops2()
        svSwaps = self.find_swaps()
        svBlocked = self.find_stop_preds(svStops.union(svSwaps))

        # Node order matters: the len(dPred)>1 branch below reads colours written by
        # earlier iterations, so we walk nodes in insertion order, as G.pred.items() did.
        for v in self.nodes:
            dPred = self.pred[v]

            if v in svSwaps:
                self.colors[v] = "purple"
            elif v in svBlocked:
                self.colors[v] = "red"
            elif len(dPred) > 1:

                if self.colors.get(v) == "red":
                    continue

                if self.agent_at.get(v) is None:
                    self.colors[v] = "blue"
                else:
                    self.colors[v] = "magenta"

                # predecessors of a contended cell
                diAgCell = {self.agent_at.get(vPred): vPred for vPred in dPred}

                # remove the agent with the lowest index, who wins
                iAgWinner = min(diAgCell)
                diAgCell.pop(iAgWinner)

                # Block all the remaining predecessors, and their tree of preds
                for iAg, vLoser in diAgCell.items():
                    for vPred in self._reverse_reachable([vLoser]):
                        self.colors[vPred] = "red"

    def check_motion(self, iAgent, rcPos):
        """ If agent position is None, we use a dummy position of (-1, iAgent)
        """

        if rcPos is None:
            rcPos = (-1, iAgent)

        # If it's been marked red or purple then it can't move
        if self.colors.get(rcPos) in ("red", "purple"):
            return (False, rcPos)

        rcNext = self.succ.get(rcPos)

        # This should never happen - only the next cell of an agent has no successor
        if rcNext is None:
            print(f"error condition - agent {iAgent} node {rcPos} has no successor")
            return (False, rcPos)

        if rcNext == rcPos:  # the agent didn't want to move
            return (False, rcNext)
        # The agent wanted to move, and it can
        return (True, rcNext)
