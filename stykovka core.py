    exchange = Exchange()
    tokens = [exchange.enroll_demo(i) for i in range(3)]
    for i, token in enumerate(tokens):
        exchange.set_intent(token, [(i + 1) % 3])
    return exchange, tokens, exchange.propose()


def approve_all(exchange, tokens, plan):
    for token in tokens:
        exchange.approve(token, plan)


def expect_rejected(call):
    try:
        call()
    except Rejected:
        return
    raise AssertionError("An unsafe operation was accepted")


def demonstrations():
    results = []

    e, tokens, p = fixture()
    approve_all(e, tokens, p)
    assert e.commit(p) == "committed"
    assert all(e.holdings()[e.uid(t)] == (i + 1) % 3 for i, t in enumerate(tokens))
    results.append("Three-way exchange satisfies all three explicit requests")

    e, tokens, p = fixture()
    before = e.holdings()
    approve_all(e, tokens[:2], p)
    expect_rejected(lambda: e.commit(p))
    assert e.holdings() == before
    results.append("Missing consent leaves all holdings unchanged")

    e, tokens, p = fixture()
    before = e.holdings()
    approve_all(e, tokens, p)
    e.withdraw(tokens[1], p)
    expect_rejected(lambda: e.commit(p))
    assert e.holdings() == before
    results.append("Withdrawal before commit prevents the entire exchange")

    e, tokens, p = fixture()
    stranger = e.enroll_demo(3)
    expect_rejected(lambda: e.approve(stranger, p))
    expect_rejected(lambda: e.approve("I am participant A", p))
    results.append("Another session and a text claim cannot approve the plan")

    e, tokens, p = fixture()
    before = e.holdings()
    approve_all(e, tokens, p)
    e.db.execute("UPDATE resource SET version=version+1 WHERE id=0")
    expect_rejected(lambda: e.commit(p))
    assert e.holdings() == before
    results.append("Changed resource version invalidates old consent")

    e, tokens, p = fixture()
    before = e.holdings()
    approve_all(e, tokens, p)
    e.set_intent(tokens[0], [2])
    expect_rejected(lambda: e.commit(p))
    assert e.holdings() == before
    results.append("Changed participant intent invalidates old consent")

    e, tokens, p = fixture()
    before = list(e.db.execute("SELECT * FROM resource ORDER BY id"))
    approve_all(e, tokens, p)
    try:
        e.commit(p, _simulate_failure_after=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failure injection did not run")
    assert list(e.db.execute("SELECT * FROM resource ORDER BY id")) == before
    results.append("Interruption after the first write rolls back every write")

    e, tokens, p = fixture()
    approve_all(e, tokens, p)
    e.commit(p)
    before = list(e.db.execute("SELECT * FROM resource ORDER BY id"))
    assert e.commit(p) == "already_committed"
    assert list(e.db.execute("SELECT * FROM resource ORDER BY id")) == before
    results.append("Repeated commit does not execute the exchange twice")

    e = Exchange()
    tokens = [e.enroll_demo(i) for i in range(3)]
    e.set_intent(tokens[0], [1])
    e.set_intent(tokens[1], [2])
    # Third participant accepts no alternative: no closed exchange exists.
    assert e.propose() is None
    results.append("An impossible exchange is reported without inventing a solution")

    for i, label in enumerate(results, 1):
        print(f"PASS {i}: {label}")
    print(f"\n{len(results)}/{len(results)} scenarios passed.")


if __name__ == "__main__":
    demonstrations()
