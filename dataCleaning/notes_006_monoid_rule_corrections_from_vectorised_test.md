during the vectorised benchmark test, two monoid rules were discovered to be incompatible with flow-level cicflowmeter data and were disabled. this is a significant architectural finding.

## rule 1: null scan (disabled)

the original rule flagged any flow where total_flags == 0 and duration > 0, under the assumption that a tcp flow with no flags is a reconnaissance null scan.

the problem: cicflowmeter produces flow records for all traffic including udp and icmp. these protocols have no tcp flags, so they naturally produce total_flags == 0. without a protocol column in the dataset, it is mathematically impossible to distinguish a tcp null scan from a legitimate udp dns query or icmp ping at the flow level.

impact: 29.92% of benign flows were being incorrectly flagged, contributing massively to the fpr of 0.71.

fix: rule disabled for flow-level data. remains valid for raw packet-level inspection where the ip protocol field is available.

## rule 6: stealth psh (disabled)

the original rule flagged any flow where psh > 0 and ack == 0, under the assumption that a data push without an acknowledgement is a malformed tcp state.

the problem: cicflowmeter counts tcp flags per flow direction independently. in a bidirectional flow, the forward direction may carry psh while the ack is in the reverse direction record. this means a perfectly normal http response can appear as psh=1, ack=0 at the flow level even though the underlying tcp session is valid.

impact: 38.6% of benign flows were being incorrectly flagged.

fix: rule disabled for flow-level data. remains valid for raw packet-level inspection where per-packet flag state is preserved.

## final benchmark results after corrections

| metric | before | after |
|---|---|---|
| fpr on benign | 0.7121 | 0.0190 |
| flows dropped at router | 36.5% | 83.8% |
| efficiency ratio | 1.58x | 6.17x |
| token savings | 36.5% | 83.8% |

the corrected monoid engine now operates on 4 rules that are valid at flow level: xmas scan, syn flood, orphaned loop, and push exhaustion. the spectral snr results are unchanged since they depend only on the svd arm.
