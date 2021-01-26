library(ordinal)
library(car)
library(RVAideMemoire)
library(emmeans)
library(stringr)

args <- commandArgs(trailingOnly = TRUE)

filename <- args[1]
score_name <- args[2]
is_crossed <- FALSE

mode_info <- unlist(str_split(args[3], ":"))

effects_mode <- mode_info[1]

if (effects_mode == "crossed") {
	is_crossed <- TRUE
}

adjust <- "tukey"
if (length(mode_info) > 1) {
	adjust <- mode_info[2]
}

outfile <- NULL
if (length(args) == 4) {
	outfile <- args[4]
}


#data <- read.csv(file="~/Documents/work/summary_repository/large_results_crossed.csv")
data <- read.csv(file=filename)

data$annotator <- factor(data$annotator)
data$document <- factor(data$document)
data$system <- factor(data$system)
#data$score <- factor(data$coherence_score, ordered=TRUE)

scores <- data[score_name]

if (score_name == "rank") {
	scores <- -scores
}

data$score <- factor(unlist(scores))

#print(head(scores))
#data$score <- factor(scores, ordered=TRUE)

#print(head(data))

if (is_crossed) {
	model <- clmm(score ~ system + (system|annotator) + (system|document), data=data)
} else {
	model <- clmm(score ~ system + (system|annotator), data=data)
}


if (!is.null(outfile)) {
	random_effects <- VarCorr(model) #lapply(model$ST, function(r) {r[upper.tri(r)] = t(r)[upper.tri(r)]; r})
	coefficients <- model$beta
	coeff_names <- unlist(lapply(names(coefficients), function(n) { substring(n, 7) }))
	system_names <- c("__REFERENCE__", coeff_names)
	thresholds <- unname(model$alpha)
	coefficients <- c(0.0, unname(coefficients))

	library(rjson)
	model_json = toJSON(list(
		random_effects=random_effects,
		coefficients=coefficients,
		system_names=system_names,
		thresholds=thresholds
	))

	write(model_json, outfile)
}


#print(summary(model))
#anova <- Anova.clmm(model)
#anova["Pr(>Chisq)"] < 0.01 || 
if (TRUE) {
	marginal <- emmeans(model, "system")
	marginal <- pairs(marginal, infer=c(TRUE, TRUE), adjust=adjust)
        marginal <- as.data.frame(marginal)

	#sig_diffs <- marginal[marginal["p.value"] < 0.05,]

	for (idx in 1:nrow(marginal)) {
		sign <- "o"
		if (marginal[idx, "p.value"] < 0.05) {
			sign <- "+"
			if (marginal[idx, "estimate"] < 0.0) {
				sign <- "-"
			}
		}
		cat(unlist(marginal[idx, "contrast"])[1])
		cat("\t")
		cat(sign)
		cat("\t")
		cat(marginal[idx, "p.value"])
		cat("\n")
	}
}

