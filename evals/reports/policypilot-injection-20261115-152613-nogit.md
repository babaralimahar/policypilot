# Injection: attack success rate

10 poisoned pages in the all-employees scope; each attack question retrieves one.

| Setting | Attacks succeeded | ASR | Blocked by output filter |
|---|---|---|---|
| Before (naive prompt, no filter) | 6/10 | 60% | - |
| After (delimited context, rules, output filter) | 0/10 | 0% | 0 |
