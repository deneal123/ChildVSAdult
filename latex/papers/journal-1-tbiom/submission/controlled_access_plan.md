# Controlled-access plan (draft pending ethics determination)

This document defines an operational access route; it does not assert that access
has been approved. The final scope must be narrowed to the ethics committee's
decision and reviewed by the institution's data-protection/legal function.

## Roles and scope

- **Data controller:** to be completed with the university's official legal entity
  and contact.
- **Access committee:** at least the principal investigator, the institutional
  data-protection contact, and one independent project member.
- **Eligible requester:** a named researcher at a verifiable research institution
  with an approved, non-commercial biometric-research protocol.
- **Permitted purpose:** reproduction of the paper's aggregate cross-age
  verification results. Identification, re-identification, surveillance,
  deployment, and model training for operational identification are prohibited.
- **Eligible data:** only the minimum fields expressly covered by the committee.
  Raw posts, captions, names, account/community identifiers, social graphs, URLs,
  and direct platform identifiers are excluded. Minors and face images are
  excluded unless the decision explicitly permits their controlled transfer.

## Request workflow

1. The requester submits identity, institution, supervisor/PI, purpose, requested
   fields, security controls, retention period, ethics decision, and evidence of
   authority to sign the DUA.
2. Two access-committee members independently record approve/reject and reasons.
   Conflicts of interest require recusal. Silence is not approval.
3. The controller and requester sign the final DUA. Access is project- and
   person-specific, time-limited, and cannot be delegated.
4. The release is built from an allowlist, encrypted for the named recipient, and
   logged by dataset version, field list, object count, checksum, recipient, date,
   and expiry. The decryption secret travels through a separate channel.
5. The requester verifies checksums and reports any mismatch before processing.

Target service levels after institutional approval: acknowledge within five
working days; decide a complete request within twenty working days; default access
term six months. These are operational targets, not promises by T-BIOM or IEEE.

## Security and audit

- Store data only on institution-managed encrypted systems with least-privilege
  access, MFA, access logs, current security updates, and encrypted backup.
- Do not upload data or derived biometric templates to public cloud services,
  public model hubs, shared notebooks, or third-party APIs.
- Maintain a named-user access log and provide it on request.
- Report suspected loss, unauthorized access, or prohibited use within 24 hours;
  stop processing until the controller closes or remediates the incident.
- Publications may contain aggregate statistics only. Any example or record-level
  output requires separate written clearance before submission.

## Withdrawal, expiry, and deletion

The controller may suspend access for a participant request, platform takedown,
committee condition, legal request, security incident, or DUA breach. On expiry or
notice, the requester stops processing and deletes the package, local derivatives,
biometric templates, and backups where technically feasible within 30 days, then
returns a signed deletion certificate. Immutable backups must be quarantined from
use and expire under the institution's documented rotation policy.

The release ledger retains administrative facts and checksums, not face data. A
data subject can use the removal contact published with the study without proving
research value or explaining the request; the controller verifies the target
privately and propagates a revocation list to active recipients.

## Release readiness checklist

- [ ] Committee decision and data-protection review are attached.
- [ ] Exact eligible fields and treatment of minors/templates are resolved.
- [ ] Request form, two-person decision record, DUA, release ledger, incident form,
      revocation notice, and deletion certificate have institutional owners.
- [ ] Package builder fails closed on filenames/fields outside the allowlist.
- [ ] A dry run proves encryption, checksum verification, revocation delivery, and
      deletion attestation without using publishable raw samples.
- [ ] T-BIOM confirms that the resulting access route meets its data policy.
