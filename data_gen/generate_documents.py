"""
Writes a small synthetic corpus of university-library-style documents to
data/documents/*.md -- invented policies/FAQs for a fictional "Whitmore
University" library, used to demo retrieval-augmented Q&A. None of this
describes a real institution.
"""
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

DOCUMENTS = {
    "borrowing-policy.md": """# Borrowing Policy

Undergraduate students may borrow up to 12 items at a time for a loan
period of 3 weeks. Postgraduate taught students may borrow up to 20 items
for 4 weeks, and postgraduate research students may borrow up to 30 items
for 8 weeks. Staff borrowing limits are set by their department liaison
librarian.

Items can be renewed online up to 3 times, provided no other user has
requested them. Overdue items accrue a fine of 10p per day per item, capped
at GBP 5 per item. Fines can be paid online via the library portal or in
person at the Welcome Desk.

Reference-only items, rare books, and items in the High Demand Collection
cannot be borrowed and must be used within the library.
""",
    "opening-hours.md": """# Opening Hours

The Main Library is open 08:00-23:00 Monday to Friday, and 10:00-20:00 on
weekends during term time. During the exam period (weeks 1-3 of January and
May), the Main Library extends to 24-hour opening Sunday through Thursday.

The Science Library is open 09:00-21:00 Monday to Friday and closed on
weekends. The Law Library is open 09:00-18:00 Monday to Friday.

All library sites are closed on public holidays and for a winter closure
between 24 December and 2 January inclusive.
""",
    "referencing-guide.md": """# Referencing and Citation Guide

The University's default referencing style is Harvard (Cite Them Right).
Some departments require a different style -- check your module handbook
before submitting coursework:

- School of Law: OSCOLA
- School of Medicine and Health Sciences: Vancouver
- School of Psychology: APA 7th edition

The library runs weekly drop-in referencing clinics on Tuesdays 13:00-14:00
in the Main Library Training Room, no booking required. Reference
management software (Zotero, EndNote, Mendeley) is supported and free to
install via the university software portal.
""",
    "interlibrary-loans.md": """# Interlibrary Loans (ILL)

If a book or article you need is not held by the library, you can request
it via Interlibrary Loan through the library portal's "Request an Item"
form. Undergraduate and postgraduate taught students get 10 free ILL
requests per academic year; postgraduate research students and staff have
unlimited requests.

Typical turnaround is 5-10 working days for UK-held items and 2-4 weeks for
items sourced from overseas. You will receive an email notification when
your item arrives, and physical items must be collected within 7 days
before the request expires.
""",
    "study-room-booking.md": """# Study Room Booking

Group study rooms can be booked online up to 7 days in advance via the
library room booking system, in slots of 1-3 hours. Rooms are for groups of
2 or more -- solo bookings may be cancelled to release capacity during busy
periods.

Silent study floors (Level 3 and Level 4 of the Main Library) do not
require booking but have a strict no-talking, no-phone-calls policy. The
Level 2 collaborative study area allows quiet conversation and does not
require booking.
""",
    "printing-and-copying.md": """# Printing, Copying and Scanning

All students receive a GBP 5 printing credit per term, topped up
automatically at the start of term. Additional credit can be purchased
online at 4p per black-and-white page and 12p per colour page.

Self-service scanners are free to use and located on every floor near the
printer stations. Copyright law permits copying up to one chapter of a book
or one article from a journal issue for personal study use -- copying
beyond this limit requires a CLA licence check, available from the
Welcome Desk.
""",
    "accessibility-services.md": """# Accessibility and Inclusive Access

The library provides assistive technology rooms on the ground floor of the
Main Library, equipped with screen readers, screen magnification software,
and adjustable-height desks. Students registered with the Disability and
Wellbeing Service can request extended loan periods, a personal book-fetching
service, and priority access to accessible study rooms.

Alternative format requests (large print, audio, accessible PDF) for core
reading list texts can be submitted via the library portal and are
typically fulfilled within 5 working days. Assistance dogs are welcome in
all library spaces.
""",
    "subject-guide-computer-science.md": """# Subject Guide: Computer Science

Key databases: ACM Digital Library, IEEE Xplore, arXiv (open access). The
Computer Science liaison librarian holds drop-in hours every Wednesday
11:00-12:00 in the Science Library.

Recommended starting points for literature reviews: ACM Computing Surveys
and IEEE Transactions journals, both accessible via the library's e-journal
portal with your university login. Past final-year project reports are
archived in the institutional repository and searchable by keyword.
""",
    "subject-guide-health-sciences.md": """# Subject Guide: Health and Life Sciences

Key databases: PubMed, CINAHL, Cochrane Library, Web of Science. Systematic
review support (search strategy design, PRISMA guidance) is available by
appointment with the Health Sciences liaison librarian -- book via the
library portal at least 5 working days in advance.

NHS Evidence and NICE guidance are freely accessible without a university
login. Bioinformatics and genomics datasets referenced in course reading
lists are mirrored on the university's HPC cluster storage where licensing
permits.
""",
    "off-campus-access.md": """# Off-Campus and Remote Access

Most e-journals, e-books, and databases are accessible off campus via the
university's single sign-on -- follow the "institutional login" link on the
publisher's site and authenticate with your university credentials. No VPN
is required for the majority of resources.

A small number of specialist databases (some legal and financial datasets)
require the university VPN. VPN client setup instructions are available
from the IT Service Desk, not the library. If off-campus access fails for a
resource you should have, report it via the library portal's "Access
Problem" form.
""",
    "dissertation-and-thesis-support.md": """# Dissertation and Thesis Support

Final-year undergraduate and postgraduate taught dissertations must be
submitted to the institutional repository on completion, alongside your
department's normal submission process. Postgraduate research theses
require binding -- the library's recommended binder offers a 48-hour
turnaround and a 10% discount for students booking via the library portal.

One-to-one dissertation research support appointments (search strategy,
managing references, avoiding accidental plagiarism) can be booked with any
liaison librarian, regardless of subject.
""",
    "academic-integrity.md": """# Academic Integrity and Plagiarism

All coursework is submitted through a text-matching similarity checker as
part of the standard submission process. A similarity score alone is not
evidence of plagiarism -- markers review flagged sections in context.

The library runs an "Avoiding Accidental Plagiarism" workshop every
Thursday 14:00-15:00 in Week 3 and Week 8 of each term, open to all
students without booking. Guidance on correct paraphrasing, quotation, and
citation practice is also available in the Referencing and Citation Guide.
""",
    "contact-and-help.md": """# Contact and Help

General enquiries: library-help@whitmore-university.example (synthetic
demo address, not a real institution). Live chat support is available via
the library portal 09:00-20:00 Monday to Friday during term time.

The Welcome Desk on the ground floor of the Main Library handles fines
payments, ID card issues, and lost property. For subject-specific research
help, contact your liaison librarian directly -- listed on each subject
guide page.
""",
}


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        (DOCS_DIR / filename).write_text(content, encoding="utf-8")
    print(f"Wrote {len(DOCUMENTS)} synthetic documents to {DOCS_DIR}")


if __name__ == "__main__":
    main()
