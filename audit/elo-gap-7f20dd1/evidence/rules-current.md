# Event constants

Participant-facing. Rendered by the site. Changing a row needs the organising team's sign-off.

| Constant | Value |
|---|---|
| Registration | open now, closes Sep 11 11:00 |
| Qualifier ladder | 4-11 September |
| Rated rounds | every hour, 08:00-22:00 from Sep 4 08:00 |
| Upload close | Sep 11 11:00 |
| Final qualification | 13-round Swiss over locked builds, Sep 11 afternoon |
| Finalist invites | Sep 11, once the final Swiss has run. Seats are confirmed by reply, first come |
| Finalists | the final Swiss decides the London field. The room holds 50. Seats fill in seed order, one per UK member of a team and at most 2 per team. Invites are issued in that order and confirmed by reply, first come, until the room is full. Live final Sep 12 at Encode Club, London |
| Prizes | £1,000 winner, £500 runner-up, £250 third, at the London final. Third is the losing semi-finalist with the better final Swiss standing. Awarded to the team, split as the team decides |
| Time control | 120s plus 0.5s per move, per side, with a 90s init budget before the clock |
| Match environment | one core of an AMD EPYC 9V74, measured at 2.60 GHz. 2 GB RAM, no network, no GPU, identical hardware |
| Environment | Python 3.12 with torch, numpy, python-chess, onnxruntime and numba preinstalled at fixed versions the docs page lists. Nothing else installs and a `requirements.txt` in the zip is ignored. Ask at hello@aichessathon.com for an addition and any grant is announced to every team |
| Pondering | your process is suspended while your opponent moves, so work between your own moves does not run and each side has the core to itself while it thinks |
| Game end | an illegal move, a crash or missing the init budget loses. Losing on time loses unless the other side has no way to mate, which draws. Both sides failing voids the game. FIDE draw rules apply. A game still running at 600 plies is drawn, and the opening position counts toward those 600 |
| Openings | every game starts from a curated opening position that is close to level. Knockout ties play each position once with each colour |
| Ranking | a standard rating fitted to your current build's results. The ladder only seeds the final Swiss |
| House bots | house bots and house engines play the ladder. The engines keep a fixed rating, which holds the scale. None can qualify |
| Qualification | only the locked-build final Swiss counts, by points. An odd field gives one team a 1-point bye |
| Tie-breaks | points, Buchholz, head-to-head, earlier final submission. A level knockout tie goes to the better Swiss standing |
| Daily Five | 6-10 September. One attempt a day, five positions, 20 minutes. One wrong move ends a position. Your five are drawn for you. Start by 23:40 London. Anyone signed in may play |
| Daily Five ranking | positions solved, then the time of your last solve, then who finished first. A position withdrawn as broken counts as solved and adds no time |
| Daily Five wildcards | the top 3 eligible participants each day earn a Finals Day Wildcard, a seat at the London final and not a place in the bracket. One per person across the five days, so places roll down. Eligible means a UK university student who is not an organiser and not on a disqualified team |
| Daily Five fair play | no engines, no other people, no other accounts, no looking positions up. Every move is timed. Invitations follow review and may include solving a position in person at the final |
| Teams | 1-3 people, one team per person. Creating, joining and leaving a team close Sep 11 11:00 |
| Eligibility | Open worldwide. A team enters the final qualification Swiss if at least one member is a UK university student. Only its UK members can take a London seat, verified before invites |
| Submissions | <= 50 MB unzipped, 10 uploads per team per day, the latest valid version plays |
| Engines | third party engines are prohibited. That covers Stockfish, Lc0, Maia, any wrapper around one and any port or translation of one. Your moves come from code you wrote. Use any AI support you like to write it, as long as the submission keeps to these rules and you can explain it when asked. An engine you wrote yourself before the event is your own code. A model is not required and a classical search is a full entry |
| Models | any network you ship is one you trained yourself. Training it on positions an existing engine labelled is allowed. Starting from a published chess network is not, so fine-tuning or re-exporting one counts as shipping it |
| Training data | unrestricted, including positions annotated by an existing engine. What ships inside the zip is what the ban covers. A shipped table that answers positions past the opening is a stored search, not training data, and counts as an engine whatever produced it. Endgame tablebases are the exception |
| Verification | every submission faces automated and human checks. Each finalist team walks through how its agent was built, and a team that ships a network shows how it was trained. Disqualification can be retroactive |
| Books and tablebases | opening books and endgame tablebases are permitted as shipped data within the 50 MB cap. `chess.polyglot` and `chess.syzygy` are in the base image |
| Code | what you ship must be source a judge can read. Everything that runs is python from your zip plus the preinstalled stack. Obfuscated agents are disqualified |
