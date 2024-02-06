HIGH_LEVEL_LABEL_DEFINITIONS = {
    "fact": "The journalist will make a factual correction or addition to the article.", 
    "style": "The journalist will make a stylistic change to the article.",
    # "None": "The journalist will not make any changes to the article."
}

DETAILED_LABEL_DEFINITIONS = {
    "Add Background": "Add additional contextualizing information to the article to help readers understand the history, geography or significance of a term, personal, place or company. Note that contextualizing information is not analysis, expectations, or projections, which would fall into the Analysis intention category.",
    "Tonal Edits": "The journalist or copy-editor made the edits due to a specific personal or artistic preference. Use your intuition here: these are usually edits that introduce punch, elegance or scenery.  These edits often also have the effect of some other edit-intention, but cannot be fully ascribed to other aims.",
    "Update Background": "The journalist updates the background information in the article.",
    "Quote Addition": "The journalist adds a quote to the article.",
    "Delete background": "The journalist removes historical or geographical context of a term, person, place or company. This is not the same as Simplification, which is more about reducing the complexity or breadth of discussion.",
    "Event update": "There is a change to some event in the world that the article covers and the article needs to be updated to reflect this. Usually, there are changes to the verbs in the article, but this can also include increased death counts, stock-market changes, etc.",
    "Quote update": "There is an addition, editing or expansion of quotes in the article. Or, a quote from one person is swapped for a quote from another. Sometimes these updates are made with other intentions (e.g. to include a punchier quote).", 
    "Event addition": "There is a new event in the world that the article covers and the article needs to be updated to reflect this.",
    "Add Analysis": "The writer adds meaningful inferences from the presented information. These can be in the form of analyses, expectations, or deeper understandings. These are usually forward-looking rather than Background information, which is usually past-looking.", 
    "Syntax Correction": "Improve grammar, spelling, or punctuation. These are strictly to correct errors in syntax, not Preferential Edits. And, they need not be adhering to a formal style-guide.", 
    "Delete Quote": "The journalist removes a quote from the article.",
    "Add Eye-witness account": "The journalist adds an eye-witness account to the article.",
    "Style-guide Edits": "Edits that are made specifically to address a formal style guide (when in doubt, defer to the Associated Press style-guide). The first version violates the style guide and the revised version fixes it.",
    "Emphasize a Point": "The journalist expands or changes the sentence to highlight a fact or argument but does not introduce additional background or analytical information.", 
    "Emphasize importance": "The sentence is moved up or down in the document in order to make the sentence more prominent, or to emphasize it's connection to the events being described in other sentences.",
    "Add Information (Other)": "The journalist adds information to the article that doesn't fit into any of the other categories.",
    "Update Analysis": "The journalist updates the analysis in the article.",
}

OTHER_LABEL_DEFINITIONS = {
    # other labels not included in the > 1000 labels set 
    "simplification": "Reduces the complexity or breadth of discussion. This edit might also remove information from the article.",
    "define term": "The author provides meaning or differentiation to a term or concept that might be unknown to the reader. Note that this intention is DIFFERENT from the Background intention, which is more about providing context, e.g. historical or geographic context for a person, company, or place.",
    "Sensitivity Consideration": "The journalist rewrote the sentence because the original version is inappropriate/ may be considered insensitive.", 
    "elaboration": "Add additional contextualizing information to the article to help readers understand the history, geography or significance of a term, personal, place or company. Note that contextualizing information is not analysis, expectations, or projections, which would fall into the Analysis intention category.",
    "source-document update": "Additional written documents have been released by a government or company that warrant inclusion/reference in the article. For example, additional information included in an SEC filing, quarterly earnings report, IPCC report, etc.",
    "correction": "There are factual errors in the original version. The new version corrects the error.",
    "additional sourcing": "The new version includes evidence of new sources for additional information, usually added for confirmation purposes. Note that this is different from Quote Update or Document Update since Additional Sourcing doesn't have to result in a new quote or document reference. Can simply be an indication that the journalist obtained new evidence.",
    "De-emphasize Importance": "The sentence is moved up or down in the document in order to make the sentence less prominent, or to de-emphasize it's connection to the events being described in other sentences.",
    "Delete Analysis": "The journalist removes analysis from the article.",
}