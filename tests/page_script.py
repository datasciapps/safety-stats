"""Run the page script of a report.html under Node, against a minimal DOM.

A browser runs the page's inline script once, on load, and an uncaught error stops it there:
every section after the failing one stays empty. ``run_page`` runs the whole script the same
way and reports the first uncaught error, what the audit trail and the footer show, and any
error raised when the page's own selection controls are changed.

The DOM is a stub with only what the page uses. A page that starts to use another DOM call
fails here with a TypeError or a ReferenceError naming it; add it to ``_DRIVER``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

# The page uses optional chaining and nullish coalescing, which Node has had since 14. Older
# Node cannot parse the page, so the test is skipped there rather than failed.
NODE_MIN_MAJOR = 18

_SCRIPT = re.compile(r"<script([^>]*)>(.*?)</script>", re.S)
_ELEMENT_ID = re.compile(r"<([A-Za-z][A-Za-z0-9-]*)\b[^>]*?\bid=\"([^\"]+)\"")

_DRIVER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

class Text {
  constructor(value) { this.parentElement = null; this.textContent = String(value); }
}
class Element {
  constructor(tagName, namespaceURI) {
    this.tagName = String(tagName).toUpperCase();
    this.namespaceURI = namespaceURI || "http://www.w3.org/1999/xhtml";
    this.childNodes = [];
    this.parentElement = null;
    this.attributes = {};
    this.style = {};
    this.listeners = {};
  }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get children() { return this.childNodes.filter(node => node instanceof Element); }
  append(...nodes) {
    for (const node of nodes) {
      this.appendChild(node instanceof Element || node instanceof Text ? node : new Text(node));
    }
  }
  appendChild(node) {
    if (!(node instanceof Element || node instanceof Text)) {
      throw new TypeError("appendChild: not a node");
    }
    if (node.parentElement) node.parentElement.removeChild(node);
    node.parentElement = this;
    this.childNodes.push(node);
    return node;
  }
  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index < 0) throw new Error("removeChild: not a child of this element");
    this.childNodes.splice(index, 1);
    node.parentElement = null;
    return node;
  }
  get textContent() { return this.childNodes.map(node => node.textContent).join(""); }
  set textContent(value) {
    for (const node of this.childNodes) node.parentElement = null;
    this.childNodes = [];
    if (value !== null && value !== undefined && value !== "") this.appendChild(new Text(value));
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  addEventListener(type, listener) {
    (this.listeners[type] = this.listeners[type] || []).push(listener);
  }
}

// The elements the page's HTML declares with an id, each in a parent of its own.
const byId = new Map();
for (const [tag, id] of input.elements) {
  const element = new Element(tag);
  element.id = id;
  new Element("div").appendChild(element);
  byId.set(id, element);
}
byId.get("inspect-safety-stats-data").textContent = input.data;
const document = {
  title: "",
  getElementById: id => byId.get(id) || null,
  createElement: tag => new Element(tag),
  createElementNS: (namespaceURI, tag) => new Element(tag, namespaceURI),
};
const quiet = () => {};
const context = vm.createContext({ document, console: { log: quiet, warn: quiet, error: quiet } });
const describe = error => (error && error.stack) || String(error);

let error = null;
try {
  vm.runInContext(input.code, context, { filename: "report.html" });
} catch (caught) {
  error = describe(caught);
}

// Change each selection control the script filled to each value it offers, as a reader would.
const eventErrors = [];
if (error === null) {
  for (const id of ["cohort-select", "measure-select", "independence-select", "group-select"]) {
    const select = byId.get(id);
    const options = select.children.filter(node => node.tagName === "OPTION");
    const values = options.map(node => node.value);
    for (const value of values) {
      select.value = value;
      for (const listener of select.listeners.change || []) {
        try {
          listener({ target: select, stopPropagation: quiet });
        } catch (caught) {
          eventErrors.push(`${id}=${value}: ${describe(caught)}`);
        }
      }
    }
  }
}

const audit = byId.get("audit").children.map(details => ({
  title: details.children[0] ? details.children[0].textContent : "",
  text: details.children.slice(1).map(node => node.textContent).join("\n"),
}));
// Every link the script made, with the address it was given, in document order.
const links = [];
const seen = new Set();
const collect = node => {
  if (!(node instanceof Element) || seen.has(node)) return;
  seen.add(node);
  if (node.tagName === "A") {
    const href = node.href === undefined ? null : String(node.href);
    links.push({ text: node.textContent, href });
  }
  node.childNodes.forEach(collect);
};
byId.forEach(collect);
process.stdout.write(JSON.stringify({
  error,
  eventErrors,
  audit,
  footer: byId.get("footer").textContent,
  links,
  texts: Object.fromEntries([...byId].map(([id, element]) => [id, element.textContent])),
}));
"""


@dataclass(frozen=True)
class AuditPanel:
    title: str
    text: str


@dataclass(frozen=True)
class PageRun:
    """What running a page's script showed."""

    # The first uncaught error, with its stack, or None when the script ran to its end.
    error: str | None
    # Errors raised by changing a selection control, one per control and value.
    event_errors: list[str]
    audit: list[AuditPanel]
    footer: str
    # Every link the script made, as (text, address); the address is None when it set none.
    links: list[tuple[str, str | None]]
    # The text each element with an id holds once the script has run, by id: the text a reader
    # sees there when the page loads, before any control is changed.
    texts: dict[str, str]


def node_or_skip() -> str:
    """The Node executable, skipping the calling test when Node is missing or too old."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, encoding="utf-8", check=False
    ).stdout.strip()
    match = re.fullmatch(r"v(\d+)\..*", version)
    if match is None or int(match.group(1)) < NODE_MIN_MAJOR:
        pytest.skip(f"Node.js {NODE_MIN_MAJOR} or newer is needed; found {version or 'none'}")
    return node


def run_page(node: str, html: str, directory: Path) -> PageRun:
    """Run the inline page script of ``html`` once, as a browser does on load."""

    scripts = _SCRIPT.findall(html)
    data = [body for attributes, body in scripts if "application/json" in attributes]
    code = [body for attributes, body in scripts if "application/json" not in attributes]
    if len(data) != 1 or len(code) != 1:
        raise AssertionError("a report page has one data script and one page script")
    markup = _SCRIPT.sub("", html)
    elements = [*_ELEMENT_ID.findall(markup), ("script", "inspect-safety-stats-data")]
    page = directory / "page.json"
    page.write_text(
        json.dumps({"data": data[0], "code": code[0], "elements": elements}),
        encoding="utf-8",
        newline="\n",
    )
    driver = directory / "driver.js"
    driver.write_text(_DRIVER, encoding="utf-8", newline="\n")
    completed = subprocess.run(
        [node, str(driver), str(page)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"the page driver failed:\n{completed.stderr}")
    result = json.loads(completed.stdout)
    return PageRun(
        error=result["error"],
        event_errors=result["eventErrors"],
        audit=[AuditPanel(title=item["title"], text=item["text"]) for item in result["audit"]],
        footer=result["footer"],
        links=[(item["text"], item["href"]) for item in result["links"]],
        texts=result["texts"],
    )
