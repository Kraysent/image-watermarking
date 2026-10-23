---
name: code-review
description: Only use this skill when explicitly asked to conduct a code review for a piece of code or a pull request.
---

Below is a set of different specific code review rules that you should check when asked for a review. Only check these rules and nothing else. Open-ended code reviews are not covered by this skill.

When conducting a code review, read the diff and check it against each rule below. Read relevant pieces of code if needed for understanding the context around the diff. Each rule consists of the slug of the rule and textual description of what specifically should be checked by the rule.

Your response should be a structured response in JSON form that you write in `.vscode/code-review.json` file in this repository.

JSON should have strictly the following form:

```json
[
    {
        "name": "typos",
        "file": "path/from/the/root/of/the/repo/file.py",
        "line_from": 10,
        "line_to": 20,
        "description": "Word 'asembly' is written with a typo, you likely meant 'assembly'"
    },
    {
        "name": "incorrect-comments",
        "file": "path/from/the/root/of/the/repo/another.py",
        "line_from": 15,
        "line_to": 21,
        "description": "The comment says that this algorithm works in O(N) but most common path according to numpy documentation is O(N^2)"
    }
]
```

You MUST adhere to this form because it will be later used by automation to create a user-friendly UI.

If the rule asks for a citation, put it in the `description` field.

To fetch the full diff use
```shell
git fetch origin && git diff origin/master...HEAD
```

Below are rules you should check when reviewing the code.

### duplicate-code

The code written is an obvious copy-pasta of the other part of the code in the diff or in other parts of the codebase. This rule only includes code that is an actual copy and paste of another part of the codebase or is very trivially extractable into functions.

### incorrect-comments

The comment used for a function, variable, or expression contradicts the content of the token it describes. This also includes docstrings or any other kind of documentation directly inside the code. Token names are not included in this rule.

### misleading-name

The name of a function, method, variable, parameter, or class contradicts what it actually does. Examples of contradictions: a get_* or fetch_* function that mutates state, an is_*/has_* name that does not return a boolean, a singular name bound to a collection, a boolean flag whose name implies the opposite polarity of the behavior it controls, or a verb that names a different operation than the one performed. Only flag when the mismatch is visible in the body of the token in the diff or in the code the diff calls. Do not flag names that are merely vague, short, or abbreviated.

### typos

A comment, token, or any other part of the code contains a typo.

### obvious-comment

A new or changed comment only restates what the next line or block already makes obvious, and adds no non-obvious rationale, constraint, or decision. Do not flag comments that explain why, trade-offs, invariants, or external constraints.

### stale-reference

The diff renames, moves, or deletes a symbol, module, file, CLI flag, config key, environment variable, or API field, but references to the old name survive elsewhere in the repository, in places the type checker does not see: comments, docstrings, error and log message strings, README or docs pages, YAML or TOML config, makefile targets, or test names. Only flag when you have located the surviving old reference and can cite its file and line.

### caller-impact

The diff changes the observable contract of a function used outside the diff, without updating those callers. Contract changes that count: the meaning or unit of a return value, None versus empty collection, ordering guarantees, which exception type is raised, whether an input argument is mutated, or the value of a default parameter. Only flag when at least one caller outside the diff is identified by file and line and would behave differently under the new contract. Do not flag changes the type checker would already reject.

### unit-mismatch

A quantity is produced in one unit, scale, or coordinate convention and consumed as if it were in another: degrees versus radians, seconds versus milliseconds, bytes versus kilobytes, zero-based versus one-based indices, or a value converted to a target unit in one branch but not another. Only flag when the source unit is determinable from the code, a column definition, a constant name, or a unit-bearing type in the diff or its immediate context.

### swallowed-error

An except clause catches an exception and then neither re-raises it, nor logs it, nor forwards it to the caller or user through the project's reporting mechanism, so the failure becomes invisible. Only flag when the handler body discards the exception entirely (pass, bare return, or assigning a fallback with no record of the failure). Do not flag handlers at documented process boundaries that log or report the exception, and do not flag handlers whose enclosing function's contract is explicitly to tolerate the failure.

### inconsistent-duplicate

The diff modifies one instance of code that is duplicated elsewhere in the repository while leaving the other instances unchanged, so the copies now disagree. Only flag when you have located the near-identical counterpart and can cite its file and line, and when the divergence affects behavior rather than only formatting.

### generic-exception

The code raises a built-in exception such as Exception, RuntimeError, or ValueError in a situation for which the project already defines a more specific exception class, or raises an exception belonging to a different layer than the one it is raised from. Only flag when the more appropriate exception class exists in the repository and is used for comparable situations elsewhere, cited by file and line.

### layer-violation

The diff introduces an import that inverts the dependency direction already established between two packages: module A imports package B, while B already imports A's package. Only flag when you can cite an existing import establishing the opposite direction. Also flag imports of code the project treats as generated or vendored being edited or wrapped in a way that a regeneration step would overwrite.

### test-name-mismatch

The name of a test describes a behavior, input, or outcome different from what the test body actually sets up and asserts. Only flag when the discrepancy is in the substance of the test, not in wording or abbreviation.

### misleading-user-text

Text that reaches an end user contradicts the code that produces or consumes it: a form field title or description, CLI help text, a task description, or an error message.

### unmanaged-resource

A resource that requires deterministic release — database connection or cursor, transaction, file, thread, executor, temporary directory, network client — is acquired in the diff without a with block or an equivalent guaranteed-cleanup path, so it leaks when an exception is raised. Only flag when the project provides a context manager or cleanup helper for that resource, cited by file and line, or when the same resource is acquired with a with block elsewhere in the codebase.

### call-in-loop

A database query, HTTP request, or other heavy I/O call is issued once per item inside a loop where the project already provides a batched or bulk equivalent. Only flag when the batched helper exists, cited by file and line, and the loop bound is data-dependent rather than a small fixed number.

### possible-sql-injection

SQL is built by interpolating or concatenating untrusted values into the query string with request parameters, user input, or other external data without using parameterized placeholders and bound arguments. Only flag when the interpolated value is not a constant or a trusted identifier whitelist, and when a parameterized API for that database driver or ORM exists in the project or standard library.

### sensitive-data-logged

A log call, print, or error report includes a secret or personally identifying value: password, API key, token, session cookie, authorization header, private key, etc. Only flag when the logged expression is clearly that sensitive value or a structure that embeds it (for example logging an entire headers dict that contains Authorization). Do not flag redacted, hashed, or truncated forms, nor logging of non-secret identifiers such as user ids or request ids.

### hardcoded-secret

A secret is committed as a string literal (API key, password, token, private key, connection string with credentials). Only flag when the value looks like a real credential, not a placeholder/test fixture.

### ssrf

An HTTP client fetches a URL taken from user/external input with no allowlist or host restriction.

### command-injection

subprocess/os.system/os.popen runs a shell with concatenated or formatted user/external input (shell=True, or a single string command). Only flag when the injected value is not a constant/whitelist.

### pointless-wrapper

A new or changed function or method only forwards to another callable with the same arguments and return value, adding no conversion, validation, defaulting, error handling, or other logic. Only flag when call sites could invoke the inner callable directly, the wrapper does not implement an interface, protocol, or abstract method, and it is not a public re-export of a private or third-party symbol.

### redundant-parameter

All callers of a new or changed function, method or class pass the same value for the parameter such that a parameter can be deleted without affecting any behaviour of all callers.
