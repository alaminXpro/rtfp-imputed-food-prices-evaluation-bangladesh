# Data sources: licences and terms of use

Retrieved on 2026-10-09 (Asia/Dhaka, UTC+6). Each URL below was fetched and read in full,
or as stated. This file reports what each source's own terms say. It is not legal advice.

- Quotes are verbatim, including the sources' own typos.
- Bengali quotes are transcribed from a scanned PDF. The English line after each one is our own
  unofficial translation. Section 21(ii) of that policy says the Bengali text prevails.
- Lines marked **[inference]** are our reading and are not stated by the source.

## Summary

| # | Source | Licence / terms stated | Zenodo archive may include |
|---|--------|------------------------|----------------------------|
| 1 | World Bank RTFP Bangladesh (`BGD_2021_RTFP_v02_M`, Microdata catalog 6164) | Open Access. CC BY 4.0 plus the World Bank's binding mediation/arbitration addition | **Raw files and derived data**, with attribution |
| 2 | WFP "Bangladesh - Food Prices" on HDX | CC BY-IGO 3.0 | **Raw files and derived data**, with credit, a licence link and changes marked. The raw file stays under CC BY-IGO 3.0 |
| 3 | BBS CPI publications (bbs.gov.bd) | **No licence stated.** The publications print a reproduction notice: no copying for commercial purposes; the data may be used and published with acknowledgement | **Download scripts only** |
| 4 | TCB daily price lists (tcb.gov.bd) | **No licence stated** | **Download scripts only** |

---

## 1. World Bank Real Time Food Prices (RTFP), Bangladesh

**Dataset identity.** Read from https://microdata.worldbank.org/index.php/api/catalog/BGD_2021_RTFP_v02_M:

- `"id": 6164`, `"idno": "BGD_2021_RTFP_v02_M"`
- title "Monthly food price estimates by product and market"
- subtitle "Bangladesh, 110 markets, 2007/01/01-2026/10/01, version 2026-10-05"
- `"data_access_type": "open"`

The numeric-id form `.../api/catalog/6164` returned HTTP 400.

**Metadata JSON**: https://microdata.worldbank.org/metadata/export/6164/json

No key containing "licen" exists anywhere in this JSON (we searched every key). The access
fields are under `study_desc.data_access.dataset_use`:

- `restrictions`: "The estimates presented in this dataset are all based on publicly-available data.
  The dataset of price estimates is published as open data."
- `disclaimer`: "The RTFP data and metadata provided are "as is" and "as available," and every effort is
  made to ensure their timeliness, accuracy, and completeness. When errors are discovered, they are
  corrected as appropriate and feasible. For details on the terms and conditions for usage of the RTFP
  database, please refer to the license details."
- `cit_req`: "Please cite this dataset as follows: Andrée, B. P. J. (2021). Monthly food price estimates by
  product and market (Version 2026-10-05). BGD_2021_RTFP_v02_M. Washington, DC: World Bank Microdata
  Library. https://doi.org/10.48529/2ZH0-JF55"

**DDI export**: https://microdata.worldbank.org/metadata/export/6164/ddi

- `<restrctn>` reads "...based on publicly-available data from the World Food Programme. The dataset of
  price estimates is published as open data."
- `<conditions>` is empty.
- Its `<citReq>` still names "Version 2024-01-25", while the JSON names 2026-10-05.
- **[inference]** The DDI export is not refreshed with each release, so citations should take the
  version string from the JSON or from each vintage's file name.

**Get Microdata page**: https://microdata.worldbank.org/index.php/catalog/6164/get-microdata

The page's access icon has CSS class `icon-da-open`. Its "Terms and Conditions" block says:

> You are free to copy, distribute, adapt, display or include the data in other products for commercial
> and noncommercial purposes at no cost subject to certain limitations summarized below.
>
> You must include attribution for the data you use in the manner indicated in the metadata included
> with the data.
>
> You must not claim or imply that The World Bank endorses your use of the data by or use The World
> Bank's logo(s) or trademark(s) in conjunction with such use.
>
> Other parties may have ownership interests in some of the materials contained on The World Bank Web
> site. For example, we maintain a list of some specific data within the Datasets that you may not
> redistribute or reuse without first contacting the original content provider, [...]
>
> The World Bank makes no warranties with respect to the data and you agree The World Bank shall not be
> liable to you in connection with your use of the data.

**Study description**: https://microdata.worldbank.org/index.php/catalog/6164

- The "Access policy" section repeats the `restrictions` text above.
- The sources list names World Food Programme (WFP) data "published in the Humanitarian Data Exchange
  (HDX)" and FAO's Food Price Monitoring and Analysis (FPMA) Tool.

**Microdata Library terms of use**: https://microdata.worldbank.org/index.php/terms-of-use

The **Open Access** tab (HTML element `id="open_access"`) says:

> Datasets and the related documentation are made available freely to users. There is no need to be
> being logged into the application.

It then repeats the block quoted above. The stricter clause on the same page, "shall not be
redistributed or sold to other individuals, institutions, or organizations without the prior written
agreement", sits in the Direct Access, Public Use and Licensed tabs (`id="direct_access"`,
`"public_use"`, `"licensed"`), not in Open Access. We confirmed this from the HTML element ids.

**Controlling agreement.** The Microdata page links to
https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets, which now redirects to
https://www.worldbank.org/ext/en/legal/terms-conditions. The datasets section is at
https://www.worldbank.org/ext/en/legal/terms-conditions/datasets, and says:

> Unless specifically labeled otherwise, these Datasets are provided to you under a Creative Commons
> Attribution 4.0 International License (CC BY 4.0), with the additional terms below. [...] When you
> download or use the Datasets, you are agreeing to comply with the terms of a CC BY 4.0 license, and
> also agreeing to the following mandatory and binding addition:

The addition provides for non-binding mediation, then arbitration. The UNCITRAL Arbitration Rules
apply if the Licensor's copyright notice designates no other rules. The page continues:

> You agree to provide attribution to The World Bank and its data providers in the following format:
> The World Bank: Dataset name: Data source (if known). When sharing or facilitating access to the
> Datasets, you agree to include the same acknowledgment requirement in any sub-licenses of the data
> that you grant, and a requirement that any sub-licensees do the same. You may meet this requirement by
> providing the uniform resource locator (URL) of these terms of use.
>
> Some datasets and indicators are provided by third parties, and may not be redistributed or reused
> without the consent of the original data provider, or may be subject to terms and conditions that are
> different from those described above. Where applicable, these conditions are included in the dataset
> or indicator metadata.

**Summary terms**: https://data.worldbank.org/summary-terms-of-use

> Unless indicated otherwise in the data or indicator metadata, you are free to copy, distribute, adapt,
> display or include the data in other products for commercial or noncommercial purposes at no cost
> under a Creative Commons Attribution 4.0 International License, with the additional terms below.

**Restricted-data list.** http://www.worldbank.org/terms-datasets-restricted redirects to
https://data.worldbank.org/restricted-data, which says:

> Where applicable, these conditions are included in the dataset or indicator metadata, and as such the
> conditions are no longer presented on this page. This page is provided for legacy purposes only.

So third-party conditions would appear in the dataset metadata. The RTFP metadata, quoted above,
states none.

**Decision: raw files and derived data may be included.** This covers all of
`data/raw/rtfp_vintages/`: 243 vintage directories, each holding one CSV, all downloaded from
`https://microdata.worldbank.org/api/downloads/download/BGD_2021_RTFP_v02_M/` (per `urls.txt`),
584 MB on disk. Conditions, as the terms state them:

1. Cite each vintage with its own version string, in the `cit_req` format.
2. Give the attribution "The World Bank: Monthly food price estimates by product and market: [source]".
3. Put the URL of the Terms of Use for Datasets in the archive README.
4. Claim no World Bank endorsement and use no World Bank logo.

The files keep CC BY 4.0 with the World Bank addition.

## 2. WFP food prices for Bangladesh (HDX)

**HDX API**: https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-bangladesh

- `license_id`: `"cc-by-igo"`
- `license_title`: `"Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO)"`
- `license_url`: `"http://creativecommons.org/licenses/by/3.0/igo/legalcode"`
- `license_other`: `null`
- `organization.title`: `"WFP - World Food Programme"`
- `title`: `"Bangladesh - Food Prices"`
- `metadata_modified`: `"2026-10-04T09:51:00.059808"`
- `dataset_source`: `"Bangladesh Milk producer's Co-Operative Union Ltd, Bangladesh via FAO: GIEWS, DAM,
  Department of Agricultural Marketing, Department of Agriculture Marketing (DAM), FPMA, WFP Market
  Data Collection, Website of Department of Agricultural Marketing"`

**HDX licences page.** https://data.humdata.org/faqs/licenses redirects to
https://docs.humdata.org/about/data-licenses, which says:

> Under the CC BY-IGO license, you are free to share (copy and redistribute the material in any medium
> or format) and or adapt (remix, transform, and build upon the material) for any purpose, even
> commercially. The licensor cannot revoke these freedoms as long as you follow the license terms. The
> license terms are that you must give appropriate credit, provide a link to the license, and indicate
> if changes were made. You may do so in any reasonable manner, but not in any way that suggests the
> licensor endorses you or your use. Additionally, you may not apply legal terms or technological
> measures that legally restrict others from doing anything the license permits. When the Licensor is
> an intergovernmental organization, disputes will be resolved by mediation and arbitration unless
> otherwise agreed.

**HDX terms of service.** https://data.humdata.org/faqs/terms redirects to
https://docs.humdata.org/about/hdx-terms-of-service, which says:

> Organizations may use HDX to share data from other sources if the applicable license allows for
> onward sharing.
>
> After downloading a public dataset, users must follow the applicable license when using and sharing
> the data.

**Licence legal code**: https://creativecommons.org/licenses/by/3.0/igo/legalcode

- Section 3: "...to Reproduce, Distribute and Publicly Perform the Work, to incorporate the Work into one
  or more Collections, [...] and, to create, Reproduce, Distribute and Publicly Perform Adaptations,
  provided that You clearly label, demarcate or otherwise identify that changes were made to the
  original Work."
- Section 4(a): "You may Distribute or Publicly Perform the Work only under the terms of this License.
  You must include a copy of, or the Uniform Resource Identifier (URI) for, this License with every copy
  of the Work You Distribute or Publicly Perform. You may not offer or impose any terms on the Work that
  restrict the terms of this License [...] You may not sublicense the Work (see section 8(a)). You must
  keep intact all notices that refer to this License and to the disclaimer of warranties with every copy
  of the Work You Distribute or Publicly Perform."
- Section 4(b) requires keeping copyright notices and giving the following credit:
  - (i) "any attributions that the Licensor indicates be associated with the Work as indicated in a
    copyright notice"
  - (ii) "the title of the Work if supplied"
  - (iii) "the URI, if any, that the Licensor specifies"
  - (iv) for Adaptations, "a credit identifying the use of the Work in the Adaptation"

**Decision: raw files and derived data may be included.** This covers
`data/raw/hdx_wfp/wfp_food_prices_bgd_2026-10-04.csv`. Conditions, as the licence states them:

1. Ship the licence URI with the raw file.
2. Keep its notices.
3. Credit WFP, the title "Bangladesh - Food Prices" and the HDX dataset URL.
4. Label derived files as changed.
5. Apply no other terms to the raw file.

**[inference]** If the Zenodo record has a single licence field set to anything else, the README must
state that this file stays under CC BY-IGO 3.0.

Upstream bodies are listed in `dataset_source`, including the Department of Agricultural Marketing
(DAM). **[inference]** Under the HDX terms quoted above, the contributing organisation (WFP) is the party
expected to confirm that onward sharing is allowed. We did not read DAM's own terms.

## 3. Bangladesh Bureau of Statistics (BBS), CPI publications

**Website.** Pages read:

- **Homepage** https://bbs.gov.bd/. The footer has no copyright or licence statement. Its only notice is:
  "এই ওয়েবসাইটে প্রকাশিত সকল তথ্য সংশ্লিষ্ট দপ্তর কর্তৃক নিয়মিত হালনাগাদ করা হয়। তথ্যের যথার্থতা,
  নির্ভুলতা ও নির্ভরযোগ্যতা নিশ্চিত করতে সংশ্লিষ্ট দপ্তর সর্বদা সচেষ্ট।"
  ("All information published on this website is regularly updated by the office concerned. The office
  concerned always strives to ensure the correctness, accuracy and reliability of the information.")
- **Policies page** https://bbs.gov.bd/pages/policies lists nothing: "কোনো তথ্য পাওয়া যায়নি।" ("No information found.")
- **CPI page** https://bbs.gov.bd/pages/static-pages/6922de7a933eb65569e1ae8f ("সিপিআই ও শিল্প উৎপাদন সূচক").
  It lists the "Monthly Release on Price & Wage Statistics: (CPI and WRI)" as PDF and Excel, and the
  rebasing books. The page states no licence.

**Notice printed in the publications.**

The monthly release for June 2026 is at
https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-bbs/2026/6/a97d7f48-8468-4232-8db9-e4beed549a34.pdf
(sha256 `6511f38d7f681d070ee5566408bccffded9c6ed536a050c45dcf9b95abdf7490`). PDF page 5 says:

> This publication or any portion thereof cannot be copied or reproduced for any commercial purpose.
> Data therein can, however, be used and published with acknowledgement of their sources.

The book "Revision and Rebasing of the Consumer price Index (CPI), 2021-2022" is at
https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-bbs/2024/12/26968b0f41e74939a1dcafcd61013df9.pdf
(sha256 `b47a63287f430d82292b28fa651956b8c0ec6ce6a58a6ea45be78e39615ac5a9`). PDF page 4 says:

> Note: This document or any portion thereof cannot be copied, microfilmed or reproduced for any
> commercial purpose, Data therein can, however, be used and published with acknowledgement of their
> sources

The Excel file for June 2026 is at
https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-bbs/2026/7/2d0a8988-310b-491a-a744-bd376783473a.xlsx.
None of its 16 sheets contains a copyright, licence or acknowledgement statement; we searched every
cell. We checked only these three files. Other months may differ.

**Data Release Policy 2026 (উপাত্ত প্রকাশ নীতিমালা, ২০২৬).**

- Issued by the Statistics and Informatics Division, Ministry of Planning, memo no.
  ৫২.০০.০০০০.০০০.০৩৩.২২.০০৫২.২৫-১২০, dated ০৯ ফেব্রুয়ারি ২০২৬, under Statistics Act 2013 s.16.
- Linked from https://bbs.gov.bd/pages/static-pages/6922e124933eb65569e2a7b3.
- PDF: https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-bbs/2026/1/ec837c29-5c57-422b-acc3-6227ee8be097.pdf
  (8 scanned pages, sha256 `26d396c75a80c52e13865f4f34f336f3e840781946ef721790942c0c7f0642f9`).
- All 8 pages were read.

The operative sentences:

- **3(i)** "এই নীতিমালা বাংলাদেশ পরিসংখ্যান ব্যুরো কর্তৃক সংগৃহীত, প্রক্রিয়াজাত ও প্রস্তুতকৃত পরিসংখ্যান এবং
  উপাত্ত প্রকাশের ক্ষেত্রে প্রযোজ্য হইবে।"
  ("This policy shall apply to the release of statistics and data collected, processed and prepared by
  BBS.")
- **10** "বিবিএস নিয়মিতভাবে মোট দেশজ উৎপাদ (GDP) প্রাক্কলন, প্রবৃদ্ধির হার (Growth Rate), ভোক্তা মূল্যসূচক (CPI),
  মূল্যস্ফীতি হার (Inflation Rate) এবং মজুরি হার সূচক (WRI) পরিমাপ পদ্ধতি ও ব্যবহৃত উপাত্তসহ পরিসংখ্যান ও তথ্য
  ব্যবস্থাপনা বিভাগের সচিবের অনুমোদনক্রমে প্রকাশ করিবে"
  ("BBS shall regularly publish GDP estimates, growth rate, CPI, inflation rate and WRI, with the
  measurement method and the data used, with the approval of the Secretary, SID.")
  This clause covers publication. It says nothing about reuse.
- **16 "সরবরাহকৃত উপাত্ত ব্যবহারের শর্তাবলি"** ("Conditions for use of supplied data"):
  - **(iv)** "বিবিএস-এর মাইক্রোডেটা ব্যবহার করিয়া প্রস্তুতকৃত প্রকাশনা, প্রতিবেদন বা গবেষণায় উৎস হিসেবে বিবিএসকে
    যথাযথ স্বীকৃতি (Acknowledgement) প্রদান করিতে হইবে।"
    ("Publications, reports or research prepared using BBS microdata must give BBS due acknowledgement
    as the source.")
  - **(vi)** "মূল্য পরিশোধ সাপেক্ষে বিবিএস হইতে প্রাপ্ত শুমারি বা জরিপের উপাত্ত বা মাইক্রোডেটা কোনো অবস্থাতেই অন্য
    কোনো ব্যক্তি, প্রতিষ্ঠান বা সংস্থাকে বিনামূল্যে বা অর্থের বিনিময়ে সরবরাহ করা যাইবে না।"
    ("Census or survey data or microdata obtained from BBS against payment may under no circumstances be
    supplied to any other person, institution or organisation, free or for money.")
- **Schedule 1 (তফসিল ১)**, row "বিনামূল্যে সর্বজনীন ব্যবহারের ডেটা":
  "বিবিএস ওয়েবসাইট http://www.bbs.gov.bd এ আপলোড করা সর্বজনীন ব্যবহারের ডেটা/রিপোর্টসমূহ বিনামূল্যে ডাউনলোড
  করা যাইবে।"
  ("Public-use data/reports uploaded to the BBS website http://www.bbs.gov.bd can be downloaded free of
  charge.")
- **18(iii)** "সরকারি তথ্য উন্মুক্তকরণ (Open Government Data - OGD) কাঠামো বাস্তবায়নের ক্ষেত্রে, যদি এই নীতিমালার
  সঙ্গে কোনো অসঙ্গতি বা সংঘর্ষ দেখা দেয়, তবে OGD নীতিমালা প্রাধান্য পাইবে।"
  ("In implementing the Open Government Data (OGD) framework, if any inconsistency or conflict with this
  policy arises, the OGD policy shall prevail.")
- **21(ii)** "এই নীতিমালার বাংলা ও ইংরেজি পাঠের মধ্যে বিরোধের ক্ষেত্রে বাংলা প্রাধান্য পাইবে।"
  ("In case of conflict between the Bengali and English texts of this policy, Bengali shall prevail.")

What the policy does and does not say:

- It grants free *download* of public-use reports.
- It names no licence for reusing or redistributing them.
- **[inference]** Section 16's clauses are worded for supplied microdata and paid data. The policy
  does not say whether Section 16 also covers free website downloads such as the CPI releases.

The homepage also announces a call for comments ("মতামত আহ্বান") on a *draft*
"খসড়া ডেটা ও মাইক্রোডেটা প্রকাশ ও প্রবেশাধিকার নীতিমালা-২০২৬". We did not read the draft.

**Decision: download scripts only. No licence is stated; do not archive BBS PDFs or Excel files.**

The printed notice ("Data therein can, however, be used and published with acknowledgement of their
sources") is what allows BBS figures to appear in the paper's tables, with acknowledgement.
**[inference]** Shipping a BBS-derived series, such as an extracted national food CPI, in the Zenodo
archive would rest on that same sentence. The default here is to regenerate such series from the
scripts. Including them is a choice for the authors.

**Authors' decision (2026-10-09):** the extracted national food CPI and the Table 7n series derived from it
are published in `data/third_party/bbs/`, on the basis of the printed notice: acknowledged to BBS, not
covered by the repository's MIT or CC BY 4.0 licences, and marked "no commercial reproduction". The raw
BBS PDF and Excel files are still not redistributed.

## 4. Trading Corporation of Bangladesh (TCB), daily price lists

Pages read:

- **Homepage** https://tcb.gov.bd/. The footer has no copyright or licence statement, only the same
  accuracy notice quoted in section 3.
- **Policies page** https://tcb.gov.bd/pages/policies (linked from the footer). It lists 5 documents:
  - চাকুরীর প্রবিধানমালা (service regulations)
  - প্রশাসনিক ও আর্থিক ক্ষমতা (administrative and financial powers)
  - তথ্য অবমুক্তকরণ নির্দেশিকা ২০১৫ (information disclosure guideline 2015)
  - রপ্তানি নীতি ২০১৫-২০১৮ (export policy 2015-2018)
  - আমদানী নীতি আদেশ ২০১৫-২০১৮ (import policy order 2015-2018)

  Judging by the titles, none is a website terms-of-use or a data licence. The 2015 disclosure
  guideline's PDF has a legacy-font text layer that could not be searched reliably. **It was not
  reviewed in full.**
- **Price pages.** None states a licence or terms:
  - https://tcb.gov.bd/pages/daily-rmps ("প্রতিদিনের খুচরা বাজার দর", daily retail prices)
  - https://tcb.gov.bd/pages/last30-rmps (last 30 days)
  - https://tcb.gov.bd/pages/monthly-rmps/ (titled "পাইকারী বাজার দর", wholesale prices)
- **One daily file**: the retail list for 2026-10-08,
  https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-tcb/2026/9/5d9482fc-f2fa-43e3-8310-98f630aac41c.xlsx.
  - Its header reads "ট্রেডিং কর্পোরেশন অব বাংলাদেশ", "www.tcb.gov.bd" and
    "ঢাকা মহানগরীর নিত্য প্রয়োজনীয় দ্রব্যের খুচরা বাজারদর (মূল্য টাকায়)।" (Dhaka city retail prices).
  - It ends with an internal distribution list.
  - No cell contains a copyright, licence, source or permission statement; we searched every cell.

**Decision: no licence stated. Download scripts only.**

---

## Not checked

- The World Bank Data Catalog entry for RTFP: its API returned HTTP 429 (rate limited).
- FAO FPMA terms (an upstream source of RTFP), WFP's data terms outside HDX, and DAM's terms.
- The Bangladesh OGD policy that BBS Data Release Policy 18(iii) refers to.
- The BBS draft Data and Microdata Release and Access Policy 2026.
- The full text of TCB's তথ্য অবমুক্তকরণ নির্দেশিকা ২০১৫.
- Any BBS or TCB page or file not named above.

Re-check every URL before the archive is deposited. Terms pages change, and HTML pages here were not
hashed because they carry dynamic content.
