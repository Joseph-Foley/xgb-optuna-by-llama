# Can Code LLama generate a data science work flow?
LLMs have taken the world by storm! I find myself using them for all sorts of things, particularly for cooking!

Now we’re seeing LLMs tailor made for specific use cases. There exist some that are adept at writing code. Github copilot and GPT-4 have been doing the rounds but now there is a new kid on the block, Code Llama from Meta.

I’ve taken an interest in Code Llama because its open source. I can freely run it on my own machine rather than rely on the cloud based closed source models. It’s no slouch either! Its performance sits just behind GPT4 on various LLM coding benchmarks ([code-llama-large-language-model-coding](https://ai.meta.com/blog/code-llama-large-language-model-coding/)) . However, that’s just for the base model from Meta. Other organizations such as Phind have fine-tuned it to the point where it can surpass GPT4 and still be significantly faster to execute ([phind-model-beats-gpt4-fast](https://www.phind.com/blog/phind-model-beats-gpt4-fast)).

It was about time I gave it a go by applying it to a typical data science work flow. I wanted to see how well it could tackle a basic EDA, modeling and deployment.

I’ll be using the model at phind.com and Telco data taken from [IBM](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/). This will be a classic customer churn problem, so we’re talking classification modelling. I didn’t want to give it something too easy, so I thought id ask it to model that data with a mix of libraries. XGBoost as the model with Optuna acting as the optimiser.

I pasted my prompts and the output into a series of Jupyter notebooks. The “raw” directory contains the exact output from the LLM, even if the code didn’t work. The “refined” directory contains notebooks where I intervened to either fix or enhance the LLM outputs. I’d say only 10 – 15% of the code is from my own interventions. That’s a pretty impressive showing from Code Llama.

![](Images/Meta.PNG?raw=true)  ![](Images/phind.PNG?raw=true)

### Conclusion

Code Llama performed extremely well. It generated the code quick and most of the time it was sufficient. However there are plenty of imperfections to speak of and I shall list them here:

1. The LLM cant actually see your data. So when it writes something that transforms, plots or makes calculations, you’ll likely have to intervene so that it suits your data.

2. Code Llama has a tendency to use old methodologies and depreciated calls. This is likely because its trained on stack overflow which has lot of old posts on it.

3. Sometimes you have to be very explicit with the prompts you give it. At which point you may as well write the code yourself since it will miss things that you take for granted. This isn’t necessarily a negative thing; you just need to think carefully about what you ask it.

4.  There is sometimes a disconnect between the code and the commentary for the code.

None of these downsides are deal breakers but here’s the best part, Code Llama is aware of its own short comings. It will ask the user for clarification or it will ask you to input the names for particular variables (e.g. columns names in your unique dataframe). It will even suggest where to take the project next to make it more complete.

Overall its an impressive showing. I’m sure in the near future we’ll even have LLMs that are aware of your data and thus keep it in mind when generating code for it.