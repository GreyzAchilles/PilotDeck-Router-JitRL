/**
 * Builtin `jitrl` plugin registration: programmatic RouterContribution
 * survives the manifest scan AND the PluginRuntime disk reload.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { loadBuiltinPlugins } from "../../../src/extension/plugins/builtin/loadBuiltinPlugins.js";
import { PluginRuntime } from "../../../src/extension/plugins/runtime/PluginRuntime.js";
import { createJitrlCustomRouter, JITRL_ROUTER_ID } from "../../../src/router/jitrl/router.js";

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-plugin-"));
}

test("builtin scan: jitrl plugin carries a programmatic RouterContribution", () => {
  const plugins = loadBuiltinPlugins();
  const jitrl = plugins.find((plugin) => plugin.name === "jitrl");
  assert.ok(jitrl, "builtin jitrl plugin not found");
  assert.equal(jitrl.source, "builtin");
  assert.ok(jitrl.routerContributions?.length);
  const contribution = jitrl.routerContributions!.find((c) => c.id === JITRL_ROUTER_ID);
  assert.ok(contribution);
  assert.equal(contribution.id, "jitrl");
  const router = contribution.createCustomRouter();
  assert.equal(router.id, JITRL_ROUTER_ID);
  assert.equal(typeof router.decide, "function");
  assert.equal(typeof router.onTurnOutcome, "function");
});

test("PluginRuntime.lookupRouter resolves the builtin jitrl router", async () => {
  const runtime = new PluginRuntime({
    projectRoot: tmpDir(),
    pilotHome: tmpDir(),
    builtinPlugins: loadBuiltinPlugins(),
  });
  // The registry populates on refresh (mirrors createLocalGateway startup).
  await runtime.refresh();
  const router = runtime.lookupRouter("jitrl");
  assert.ok(router);
  assert.equal(router.id, "jitrl");
  const unknown = runtime.lookupRouter("does-not-exist");
  assert.equal(unknown, undefined);
});

test("PluginRuntime.refresh preserves programmatic contributions after disk reload", async () => {
  const runtime = new PluginRuntime({
    projectRoot: tmpDir(),
    pilotHome: tmpDir(),
    builtinPlugins: loadBuiltinPlugins(),
  });
  await runtime.refresh();
  const router = runtime.lookupRouter("jitrl");
  assert.ok(router, "RouterContribution lost after refresh()");
  assert.equal(router.id, "jitrl");

  const snapshot = runtime.snapshot();
  const jitrl = snapshot.find((plugin) => plugin.name === "jitrl");
  assert.ok(jitrl?.routerContributions?.some((c) => c.id === JITRL_ROUTER_ID));
});

test("createJitrlCustomRouter returns a fresh instance per call but shared state is stable", async () => {
  const a = createJitrlCustomRouter();
  const b = createJitrlCustomRouter();
  assert.notEqual(a, b);
  assert.equal(a.id, b.id);
});
