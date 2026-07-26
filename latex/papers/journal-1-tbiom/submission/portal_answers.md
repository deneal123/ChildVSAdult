# Ответы в форме подачи T-BIOM — готовые формулировки

Всё ниже согласовано с текстом рукописи (разд. «Ethical Approval and Legal Basis» и
«De-identification and Release Policy») и с сопроводительным письмом. Требование IEEE: то, что
написано в поле формы, **должно также присутствовать в самой статье** — оно присутствует.

---

## 1. «Did this research involve human subjects dataset collected during this research?»

**Ответ: Yes.**

Почему не «Not applicable»: мы действительно собрали в ходе исследования набор изображений лиц
живых людей. Вариант «This submission does not include human or animal research» здесь был бы
попыткой уйти от вопроса, а не честным ответом. Отвечаем «да» и объясняем в следующем поле.

## 2. «Was approval obtained from a relevant review board (or local/regional equivalent)?»

**Ответ: «Not applicable. This research is exempt.»** (второй из двух доступных вариантов —
первый, «Yes», требует ссылки на уже полученное одобрение, которого у нас пока нет).

### Текст для поля обоснования — вставить как есть

> This study was not conducted under institutional review board oversight, and we state the reason
> here, as IEEE policy permits in place of an approval reference (PSPB Operations Manual 8.1.1.B/E:
> "or include an explanation as to why such a review was not conducted").
>
> The research involves no interaction and no intervention with any person. It is a retrospective
> analysis of photographs that the subjects had themselves already published on public community
> walls of a social network, retrieved through the platform's official API. No individual is
> identified, contacted, profiled or ranked; no model or decision system is deployed; minors are
> excluded as subjects. No released artifact contains raw face images, raw posts or recoverable
> identifiers: the public release is limited to code, configurations, a de-identified pair protocol
> using salted identifier hashes, and aggregate metrics. Trained weights and embeddings are not
> released publicly, because a face embedding is itself a comparable biometric template; they are
> shared with bona-fide researchers on request under a research-use agreement.
>
> On the usual criteria for secondary analysis of already-public material without intervention, we
> consider the study exempt from prospective review. We state the counter-consideration rather than
> resolving it in our own favour: the images are biometric and the individuals in them remain
> identifiable, which is why our release policy is restrictive rather than open.
>
> Consent: because the data are drawn from public posts with no feasible channel to contact the
> subjects, individual informed consent for participation and for publication was not obtained. The
> manuscript explains this, as IEEE policy likewise permits ("or explain why consent was not
> obtained"), and relies on the scientific-research basis (GDPR Art. 6(1)(f) and 9(2)(j) with Art. 89
> safeguards; analogous research handling under Russia's 152-FZ) together with the de-identification
> safeguards described in the manuscript.
>
> A formal determination on exemption has been requested from the ethics committee of National
> Research University Higher School of Economics (HSE University), Moscow, Russia. It will be
> forwarded to the Editor, together with its filing date, as soon as it is issued, and that
> determination — not the authors' own reading — governs.
>
> Institution: National Research University Higher School of Economics (HSE University), Moscow,
> Russia. Ethics body: HSE University ethics committee. Status as of submission: exemption
> determination requested, decision pending; the committee's document will be supplied on request.

## 3. Conflict of interest

**Ответ: «None of the authors have a conflict of interest to disclose.»**
Плюс загрузить `coi.pdf` в обязательный слот. Та же формулировка стоит в разделе статьи
«Funding and Conflicts of Interest» — три места не должны расходиться.

## 4. Previously published

**Ничего не загружать.** Работа не является расширением *опубликованной* конференционной статьи.
Более ранняя версия подавалась в другой журнал IEEE и была **отклонена** — отклонённая рукопись не
считается опубликованной, поэтому слоты «Previously Published» не применяются. Факт всё равно
раскрыт в сопроводительном письме (п. 3), чтобы редактор узнал об этом от нас, а не со стороны.

## 5. Дополнительно — не спрашивают, но проверьте

- **AAAI-версия не должна быть на рецензии** ни в один момент, пока идёт эта подача (см.
  SUBMISSION.md, п. 3).
- **Даты запроса нигде нет — и это сознательно.** Дата в опубликованной статье необратима: если
  руководитель подаст запрос другим числом, расхождение с документом уже не исправить. Вместо этого
  и статья, и письмо обещают предоставить **сам документ комиссии вместе с его датой**. Если
  редактор попросит дату — она берётся из документа.
- **Единственное требование к очерёдности:** фраза «определение запрошено» должна быть истинной в
  момент нажатия «submit». То есть запрос в комиссию подаётся **до** подачи статьи, даже если
  решение придёт много позже. Порядок «отправил научруку → он подал в комиссию → подаём статью».
