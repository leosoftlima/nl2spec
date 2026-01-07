from nl2spec.comparator.mop_to_ir import mop_text_to_ir

def test_parse_fsm():
    mop = """
    Socket_SetTimeoutBeforeBlockingOutput(Socket sock, OutputStream output) {
      event enter before(OutputStream output) :
        call(* OutputStream+.write(..)) && target(output) {}

      event leave after(OutputStream output) :
        call(* OutputStream+.write(..)) && target(output) {}

      fsm :
        start [
          enter -> blocked
        ]
        blocked [
          leave -> start
        ]

      @fail {
        RVMLogging.out.println(Level.CRITICAL, "timeout missing");
      }
    }
    """
    ir = mop_text_to_ir(mop, spec_id="x")
    assert ir["category"] == "FSM"
    assert ir["ir"]["type"] == "fsm"
    assert len(ir["ir"]["transitions"]) > 0
