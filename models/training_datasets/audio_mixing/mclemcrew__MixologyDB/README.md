---
license: mit
language:
- en
tags:
- music
size_categories:
- n<1K
---

**Motivation for Dataset Creation**
- *Why was the dataset created? (e.g., were there specific tasks in mind, or a specific gap that needed to be filled?)*
This dataset was created to help advance the field of intelligent music production, specifically targeting music mixing in a digital audio workstation (DAW).

- *What (other) tasks could the dataset be used for? Are there obvious tasks for which it should not be used?*
This dataset could possibly be used to predict parameter values via semantic labels provided by the mixed listening evaluations.

- *Has the dataset been used for any tasks already? If so, where are the results so others can compare (e.g., links to published papers)?*
Currently, this dataset is still being curated and has yet to be used for any task. This will be updated once that has changed.

- *Who funded the creation of the dataset? If there is an associated grant, provide the grant number.*
The National Science Foundation Graduate Research Fellowship Program (Award Abstract # 1650114) helped to financially support the creation of this dataset by helping financially support the creator through their graduate program.

- *Any other comments?* 

**Dataset Composition**
- *What are the instances? (that is, examples; e.g., documents, images, people, countries) Are there multiple types of instances? (e.g., movies, users, ratings; people, interactions between them; nodes, edges)*
The instances themselves are annotated of individual mixes either from Logic Pro, Pro Tools, or Reaper, depending on the artist who mixed them.

- *Are relationships between instances made explicit in the data (e.g., social network links, user/movie ratings, etc.)?*
Each mix is unique to the other, and there exists no evident relationship between them.

- *How many instances of each type are there?*
There will be 114 mixes once this dataset is finalized.

- *What data does each instance consist of? "Raw" data (e.g., unprocessed text or images)? Features/attributes? Is there a label/target associated with instances? If the instances are related to people, are subpopulations identified (e.g., by age, gender, etc.) and what is their distribution?*
Each instance of a mix contains the following: Mix Name, Song Name, Artist Name, Genre, Tracks, Track Name, Track Type, Track Audio Path, Channel Mode, Parameters, Gain, Pan, (Etc)

- *Is everything included or does the data rely on external resources? (e.g., websites, tweets, datasets) If external resources, a) are there guarantees that they will exist, and remain constant, over time; b) is there an official archival version. Are there licenses, fees or rights associated with any of the data?*
The audio that is associated with each mix is an external resource, as those audio files are original to their source. The original audio sources are from The Mixing Secrets, Weathervane, or The Open Multitrack Testbed.

- *Are there recommended data splits or evaluation measures? (e.g., training, development, testing; accuracy/AUC)*
There are no data splits recommended for this. However, suppose no listening evaluation is available for that current mix. In that case, we recommend leaving out that mix if you plan on using those comments for the semantic representation of the mix. All of the mixes that were annotated from Mike Senior's The Mixing Secret projects for sound on sound do not contain any listening evaluation.

- *What experiments were initially run on this dataset? Have a summary of those results and, if available, provide the link to a paper with more information here.*
No experiments have been run on this dataset as of yet.

- Any other comments? 

**Data Collection Process** 
- *How was the data collected? (e.g., hardware apparatus/sensor, manual human curation, software program, software interface/API; how were these constructs/measures/methods validated?)*
The data was collected manually by annotating parameter values for each track in the mix. The mix projects were provided as Logic Pro, Pro Tools, or Reaper files. Each project was opened in their respective software and the author went through each track and annotated these parameters manually.  A tool was created to help assemble this dataset for parameter values that plugin manufacturers obscured.  This tool estimated the value of each parameter based on the visual representation that was provided in the plugin.

- *Who was involved in the data collection process? (e.g., students, crowdworkers) How were they compensated? (e.g., how much were crowdworkers paid?)*
The author of this dataset collected the data and is a graduate student at the University of Utah.

- *Over what time-frame was the data collected? Does the collection time-frame match the creation time-frame?*
This dataset has been collected from September through November of 2023.  The creation time frame overlaps the collection time frame as the main structure for the dataset was created, and mixes are added iteratively.

- *How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part of speech tags; model-based guesses for age or language)? If the latter two, were they validated/verified and if so how?*
The data were directly associated with each instance.  The parameter values are visually represented in each session file for the mixes.

- *Does the dataset contain all possible instances? Or is it, for instance, a sample (not necessarily random) from a larger set of instances?*
The dataset contains all possible instances that were given by The Mix Evaluation Dataset, negating the copyrighted songs that were used in the listening evaluation.

- *If the dataset is a sample, then what is the population? What was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? Is the sample representative of the larger set (e.g., geographic coverage)? If not, why not (e.g., to cover a more diverse range of instances)? How does this affect possible uses?*
This dataset does not represent a sample of a larger population and thus, a sample size is not appropriate in this case.

- *Is there information missing from the dataset and why? (this does not include intentionally dropped instances; it might include, e.g., redacted text, withheld documents) Is this data missing because it was unavailable?*
Not all of the parameter values for every plugin used were documented.  Occasionally a mix would include a saturator or a multiband compressor.  Due to the low occurrence of these plugins, these were omitted for the annotating process.

- *Are there any known errors, sources of noise, or redundancies in the data?*
To the author's knowledge, there are no errors or sources of noise within this dataset.

- *Any other comments?*

**Data Preprocessing** 
- *What preprocessing/cleaning was done? (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values, etc.)*
The data preprocessing happened during the data collection stage for this dataset. Some of the data values were not available from the plugins that were used in a DAW session file. To help estimate the values on each of the parameters for that respective plugin, a tool was created and used by this author. If there wasn't a value for the parameter, the value was omitted from the data collection.

- *Was the "raw" data saved in addition to the preprocessed/cleaned data? (e.g., to support unanticipated future uses)*
The raw data is still saved in the project files but was not annotated and, therefore, is not contained in this dataset. For the raw files of each mix, the reader should explore The Mix Evaluation dataset for these values.

- *Is the preprocessing software available?*
The tool that was used to help the author annotate some of the parameter values is available for download [here](https://github.com/mclemcrew/MixologyDB)

- *Does this dataset collection/processing procedure achieve the motivation for creating the dataset stated in the first section of this datasheet?*
The authors of this dataset intended to create an ethical source repository for AI music researchers to use for music mixing. We believe by using The Mix Evaluation dataset along with publically available music mixing projects, we have achieved our goal. Although this dataset is considerably smaller than what is required for most model architectures utilized in generative AI applications, we hope this is a positive addition to the field.

- *Any other comments?*

**Dataset Distribution** 
- *How is the dataset distributed? (e.g., website, API, etc.; does the data have a DOI; is it archived redundantly?)*
This dataset is distributed via HuggingFace and will continue to be hosted there for the foreseeable future. There are no current plans to create an API, although a website for the dataset has been mentioned. The data is currently being archived redundantly through the University of Utah's Box account. Should HuggingFace go down or remove the dataset, the data themselves will remain at the University of Utah and will be uploaded to a separate website.

- *When will the dataset be released/first distributed? (Is there a canonical paper/reference for this dataset?)*
The dataset, in its entirety, will be released on December 5th, 2023.

- *What license (if any) is it distributed under? Are there any copyrights on the data?*
The license will be distributed via the MIT license. There are no copyrights on this data.

- *Are there any fees or access/export restrictions?*
There are no fees or access/export restrictions for this dataset.

- *Any other comments?*

**Dataset Maintenance** 
- *Who is supporting/hosting/maintaining the dataset? How does one contact the owner/curator/manager of the dataset (e.g. email address, or other contact info)?*
HuggingFace is currently hosting the dataset and Michael Clemens (email: michael.clemens at utah.edu) is maintaining the dataset.

- *Will the dataset be updated? How often and by whom? How will updates/revisions be documented and communicated (e.g., mailing list, GitHub)? Is there an erratum?*
The release of this dataset is set to be ***December 5th, 2023***.  Updates and revisions will be documented through the repository through HuggingFace.  There is currently no erratum, but should that be the case, this will be documented here as they come about.

- *If the dataset becomes obsolete how will this be communicated?*
Should the dataset no longer be valid, this will be communicated through the ReadMe right here on HF.
  
- *Is there a repository to link to any/all papers/systems that use this dataset?*
There is no repo or link to any paper/systems that use the dataset.  Should this dataset be used in the future for papers or system design, there will be a link to these works on this ReadMe, or a website will be created and linked here for the collection of works.

- *If others want to extend/augment/build on this dataset, is there a mechanism for them to do so? If so, is there a process for tracking/assessing the quality of those contributions. What is the process for communicating/distributing these contributions to users?*
This dataset is an extension of The Mix Evaluation Dataset by Brecht De Man et al., and users are free to extend/augment/build on this dataset.  There is no trackable way currently of assessing these contributions.
- 
- *Any other comments?*

**Legal & Ethical Considerations**
- *If the dataset relates to people (e.g., their attributes) or was generated by people, were they informed about the data collection? (e.g., datasets that collect writing, photos, interactions, transactions, etc.)*
As this was a derivative of another work that performed the main data collection, the original music producers who mixed these tracks were not informed of the creation of this dataset.

- *If it relates to other ethically protected subjects, have appropriate obligations been met? (e.g., medical data might include information collected from animals)*
N/A

- *If it relates to people, were there any ethical review applications/reviews/approvals? (e.g. Institutional Review Board applications)*
As this is an extension of the main dataset by Brecht De Man et al. and the data collection had already been conducted, an IRB was not included in this creation of this dataset. The data themselves are not related to the music producers but instead remain as an artifact of their work. Due to the nature of these data, an IRB was not needed.
 
- *If it relates to people, were they told what the dataset would be used for and did they consent? What community norms exist for data collected from human communications? If consent was obtained, how? Were the people provided with any mechanism to revoke their consent in the future or for certain uses?*
N/A

- *If it relates to people, could this dataset expose people to harm or legal action? (e.g., financial social or otherwise) What was done to mitigate or reduce the potential for harm?*
The main initiative of this work was to create a dataset that was ethically sourced for parameter recommendations in the music-mixing process.  With this, all of the data found here has been gathered from publically avaiable data from artists. Therefore no copyright or fair use infringement exists.
 
- *If it relates to people, does it unfairly advantage or disadvantage a particular social group? In what ways? How was this mitigated? If it relates to people, were they provided with privacy guarantees? If so, what guarantees and how are these ensured?*
N/A

- *Does the dataset comply with the EU General Data Protection Regulation (GDPR)? Does it comply with any other standards, such as the US Equal Employment Opportunity Act? Does the dataset contain information that might be considered sensitive or confidential? (e.g., personally identifying information)*
To the authors' knowledge, this dataset complies with the laws mentioned above.
 
- *Does the dataset contain information that might be considered inappropriate or offensive?*
No, this dataset does not contain any information like this.

- *Any other comments?*