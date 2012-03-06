#!/bin/bash
set -eu
files=$*

if [ -z "$files" ]; then
	echo "No file names were provided, trying all non-binary files under git control"
	files=$(git grep -l . | grep -v -e Binary -e refactor_ -e Changelog)
	if [ -z "$files" ]; then
		echo "No files were found under git, exiting"
		exit 1
	fi
fi

echo "Replacing known patters"

echo \
"absminDistance	absmin_distance
_accessKohonen	_access_kohonen
aggregateFeatures	aggregate_features
asDescreteTime	as_descrete_time
asstring	as_string
autoNullDist	auto_null_dist
auxBasic	aux_basic
buildStreamlineThings	build_streamline_things
buildVectorBasedPM	build_vector_based_pm
cartesianDistance	cartesian_distance
_checkRange	_check_range
checkRange	check_range
_checkValues	_check_values
_checkVersion	_check_version
chirpLinear	chirp_linear
_closeOpenedHandlers	_close_opened_handlers
coarsenChunks	coarsen_chunks
collectNoseTests	collect_nose_tests
collectTestSuites	collect_test_suites
compareBibByAuthor	compare_bib_by_author
compareBibByDate	compare_bib_by_date
_computeInfluenceKernel	_compute_influence_kernel
_computeRecon	_compute_recon
convert2SVMNode	seq_to_svm_node
_customizeDoc	_customize_doc
_debugCallback	_debug_callback
_demeanData	_demean_data
doubleArray2List	double_array_to_list
doubleArray	double_array
doubleGammaHRF	double_gamma_hrf
dumbFeatureBinaryDataset	dumb_feature_binary_dataset
dumbFeatureDataset	dumb_feature_dataset
enhancedDocString	enhanced_doc_string
FirstAxisMean	first_axis_mean
FirstAxisSumNotZero	first_axis_sum_not_zero
formatAuthor	format_author
formatProperty	format_property
formatSurname	format_surname
__forwardMultipleLevels	__forward_multiple_levels
__forwardSingleLevel	__forward_single_level
freeDoubleArray	free_double_array
freeIntArray	free_int_array
generateFromXML	from_xml
generateSummary	get_summary
getAsDType	get_as_dtype
_getAttrib	_get_attrib
_getBMU	_get_bmu
getBreakPoints	get_break_points
_getCvec	_get_cvec
getData	get_data
getDataT	get_data_t
_getDefaultC	_get_default_c
getEV	get_ev
_getFeatureIds	_get_feature_ids
getFullMatrix	get_full_matrix
_getGroup	_get_group
_getHandlers	_get_handlers
_getIndexes	_get_indexes
getLabels	get_labels
getLabels_map	get_labels_map
_getLevelsDict	_get_levels
_getLevelsDict_virtual	_get_levels_virtual
_getLevels	_get_selected_levels
getMajorityVote	get_majority_vote
getMap	get_map
getMaps	get_maps
getMetric	get_metric
getMVPattern	get_mv_pattern
getNeighbor	get_neighbor
_getNElements	_get_n_elements
_getNLevels	_get_n_levels
_getNLevelsVirtual	_get_n_levels_virtual
getNRClass	get_nr_class
getNSV	get_n_sv
_getRecon	_get_recon
getRho	get_rho
getSamplesPerChunkLabel	get_samples_per_chunk_target
getSensitivityAnalyzer	get_sensitivity_analyzer
_getSplitConfig	_get_split_config
getSVCoef	get_sv_coef
getSV	get_sv
getSVRPdf	get_svr_pdf
getSVRProbability	get_svr_probability
getTotalNSV	get_total_n_sv
getTriangle	get_triangle
_getUniqueLengthNCombinations_binary	unique_combinations
getUniqueLengthNCombinations	unique_combinations
_getUniqueLengthNCombinations_lt3	unique_combinations
getVectorForm	get_vector_form
getWeightedVote	get_weighted_vote
GrandMean	grand_mean
indentDoc	indent_doc
intArray2List	int_array_to_list
intArray	int_array
inverseCmap	inverse_cmap
isInVolume	is_in_volume
isSorted	is_sorted
isTrained	is_trained
joinAuthorList	join_author_list
L1Normed	l1_normed
L2Normed	l2_normed
labelPoint	label_point
labelVoxel	label_voxel
leastSqFit	least_sq_fit
levelsListing	levels_listing
levelType	level_type
loadAtlas	load_atlas
_loadData	_load_metadata
_loadFile	_load_file
_loadImages	_load_images
mahalanobisDistance	mahalanobis_distance
makeFlobs	make_flobs
manhattenDistance	manhatten_distance
matchDistribution	match_distribution
meanPowerFx	mean_power_fx
MNI2Tal_Lancaster07FSL	mni_to_tal_lancaster07_fsl
MNI2Tal_Lancaster07pooled	mni_to_tal_lancaster07pooled
MNI2Tal_MeyerLindenberg98	mni_to_tal_meyer_lindenberg98
MNI2Tal_YOHflirt	mni_to_tal_yohflirt
multipleChunks	multiple_chunks
myFirstPage	my_first_page
myLaterPages	my_later_pages
Nlevels	nlevels
normalFeatureDataset	normal_feature_dataset
oneMinusCorrelation	one_minus_correlation
OneMinus	one_minus
parsedCoordinatesIterator	parsed_coordinates_iterator
parseStatus	parse_status
percentCorrect	percent_correct
plotBars	plot_bars
plotDatasetChunks	plot_dataset_chunks
plotDecisionBoundary2D	plot_decision_boundary_2d
plotDistributionMatches	plot_distribution_matches
plotERP	plot_erp
plotERPs	plot_erps
plotErrLine	plot_err_line
plotFeatureHist	plot_feature_hist
plotHeadOutline	plot_head_outline
plotHeadTopography	plot_head_topography
plotMRI	plot_lightbox
plotProjDir	plot_proj_dir
plotSamplesDistance	plot_samples_distance
predictProbability	predict_probability
predictValues	predict_values
predictValuesRaw	predict_values_raw
prepParser	prep_parser
presentLabels	present_labels
pureMultivariateSignal	pure_multivariate_signal
_pythonStepwiseRegression	_python_stepwise_regression
RankOrder	rank_order
registerMetric	register_metric
removeInvariantFeatures	remove_invariant_features
__resetChangedData	__reset_changed_data
reuseAbsolutePath	reuse_absolute_path
ReverseRankOrder	reverse_rank_order
__reverseSingleLevel	__reverse_single_level
RFEHistory2maps	rfe_history_to_maps
ROCs	rocs
rootMeanPowerFx	root_mean_power_fx
runNoseTests	run_nose_tests
runTests	run_tests
SecondAxisMaxOfAbs	REFACTOR_USE_FXMAPPERS
SecondAxisMean	REFACTOR_USE_FXMAPPERS
SecondAxisSumOfAbs	REFACTOR_USE_FXMAPPERS
selectOut	select_out
selectSamples	select_samples
selectVoxelsFromVolumeIteratorNumPY	select_from_volume_iterator
setActiveFromString	set_active_from_string
_setActive	_set_active
_setAnalyzers	_set_analyzers
_setClassifier	_set_classifier
_setClassifiers	_set_classifiers
setCoordT	set_coordT
setDistance	set_distance
_setFElements	_set_f_elements
_setHandlers	_set_handlers
_setIndent	_set_indent
setLabels_map	set_labels_map
_setLevel	_set_level
_setMaxCount	_set_max_count
_setMode	_set_mode
_setNElements	_set_n_elements
setNPerLabel	set_n_per_label
_setOffsetByDepth	_set_offset_by_depth
_setParameter	_set_parameter
_setPrintsetid	_set_printsetid
setReferenceLevel	set_reference_level
_setRetrainable	_set_retrainable
_setStrategy	_set_strategy
_setTail	_set_tail
setTestDataset	set_test_dataset
singleGammaHRF	single_gamma_hrf
sinModulated	sin_modulated
smoothRsT	smooth_rst
spaceFlavor	space_flavor
splitDataset	split_dataset
substractBaseline	substract_baseline
Tal2MNI_Lancaster07FSL	tal_to_mni_lancaster07_fsl
Tal2MNI_Lancaster07pooled	tal_to_mni_lancaster07pooled
Tal2MNI_YOHflirt	tal_to_mni_yohflirt
testAllDependencies	check_all_dependencies
_testCompareToOld	_test_compare_to_old
__testFSPipelineWithAnalyzerWithSplitClassifier	__test_fspipeline_with_split_classifier
__testMatthiasQuestion	__test_matthias_question
_testOnSwaroopData	_test_on_swaroop_data
toEvents	to_events
toRealSpace	to_real_space
toVoxelSpace	to_voxel_space
transformWithBoxcar	transform_with_boxcar
_verboseCallback	_verbose_callback
__wasDataChanged	__was_data_changed
_waveletFamilyCallback	_wavelet_family_callback
nperlabel	npertarget
roisizes	roi_sizes
xuniqueCombinations	xunique_combinations
|model='linear'	polyord=1
|baselinetargets=	param_est=('targets'\\\,) 
targetdtype	dtype
detrend	poly_detrend
nifti_dataset	fmri_dataset"  | \
while read old new; do
	echo -en "\r$old                             "

	# def definition
	grep -l "def *$old" $files | xargs -r sed -i -e "s,^\( *\)\(def  *$old*\)(,\1##REF: Name was automagically refactored\n\1def $new(,g"
	# occurances
	if [ "${old:0:1}" = '|' ]; then
		old="${old:1}"
		# we got a complete regexp -- no need to guard
		grep -l "$old" $files | xargs -r sed -i -e "s,$old,$new,g" && echo -n "" || :
	else
		# we got a word expression -- need to guard on the boundaries
		grep -l "\<$old\>" $files | xargs -r sed -i -e "s,\<$old\>,$new,g" && echo -n "" || :
	fi
done
echo -e "\rDONE                              "
