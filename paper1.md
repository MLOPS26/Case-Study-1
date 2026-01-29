Members’ names (1 point)
Artem Frenk, Karish Gupta, Connor Jason

– Description of the product(s), their purpose, and target audience (4 points)
Our product is MathHelper, a Vision-Language Model (VLM) designed to help users solve math problems by simply taking and uploading a picture of the problem and answering questions about what it sees. The main purpose of our application is to help students or other users with math problems in a new, easier way. Opening the camera app on your phone and taking a picture of a problem is a lot easier than trying to type it out into a calculator. Our target audience is students or tutors, but it can also benefit anyone who has a visual math problem that needs to quickly and easily be solved.

– Description of the models being used, including their architecture, purpose, and any datasets
involved (4 points)
We are using Qwen2-VL-2B-Instruct to power the brain behind our application. This model is a small VLM with 2 billion parameters made by Alibaba Cloud. It combines computer vision and natural language processing to allow the model to accept images of a math problem and respond to text questions about that problem. This is a transformer-based multimodal model with a highly optimized vision transformer (ViT). This model in specific features Multimodal Rotary Position Embedding (M-ROPE). This architecture <artem u can probably talk about this better than i can> "Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities." "Naive Dynamic Resolution: Unlike before, Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience." quotes are from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct if these are useful to u
We fine-tuned this model using the MathVista dataset, which is a visual reasoning dataset of mathematical problems. This dataset is designed to help train models just like ours, making it a very strong choice for our fine-tuning.

– Performance analysis of the products (e.g., how long do they take to answer, resource usage) (3
points)
We should measure 1. how long it takes to process an image and generate a response 2. cpu usage in spaces 3. accuracy of the model
"This model runs on X CPU and averages around Y% utilization during inference."

– Cost based analysis of the products (e.g., estimated cost if you have 1,000 users) (3 points)
∗ See Hugging Face – Pricing for details on the costs associated with hosting and using models
on Hugging Face Spaces.
∗ Remember that, while not explicitly defined, rate limits apply to the Inference API usage,
so consider these limits when estimating costs for a large number of users.
1,000 users is not that many, idk how resource hungry this model is but free tier might be fine
if not, we can look at https://huggingface.co/pricing. if we have to do a team plan then the $20 is likely fine. if not, the $9 personal would be fine. A GPU is really all we need resource-wise for more requests imo
If our application were to reach 1,000 users, the CPI basic (free) tier would likely not be sufficient. Assuming our 1,000 users would be spread throughout the day (not concurrent), we would likely do well with the 8x ZeroGPU usage found in the cheapest organization paid plan, the $20/month Team plan. This plan provides dynamic GPU access with higher quotas than their free tier. This would be significantly cheaper than the next step up, a dedicated Nvidia T4 GPU. If our application outgrows the Team plan, this GPU would cost us $0.40/hour x 24 hours x 30 days = ~$288/month. This is significantly more than the first suggestion, but would make it possible to serve sustained high traffic, ensure consistent performance, and avoid rate limiting.

– Comments and/or concerns, such as potential security issues, data privacy, and scalability (3
points)
* i have no idea how HF does things plz correct me if im wrong im literally guessing
Since users are uploading their own images, we should be sure to not save these images long-term and ensure that humans cannot view the images users upload for their privacy. As our app grows, we will be limited by the inference of the model. For example, even if the model takes just five seconds to make an inference, this would slow our application down and eventually bring it to a halt with anything more than one request every five seconds. Eventually, the queue would back up and we would be forced to drop requests or otherwise prevent usage of our application. This model is rather small, so if we get a decent amount of resources, we should be able to scale up to a few instances of the model to run multiple inferences in parallel. That way, we can distribute these requests to whichever model is free/has the least amount of work, keeping responses to our users quick. Outside of this, I wouldn't expect our code to need much altering.

– Additional insights, challenges faced, and potential future improvements (2 points)
For future improvements, we would like to fine-tune our model on more diverse datasets to improve the accuracy of our model. We would also like to add an API for our more technically-savvy users to access, should they want to. This would require rate limiting, API keys, etc, so for now it is considered future work. And, if feasible, we could cache responses for common questions. I suspect the problems our users will be rather diverse, but it's usually worth considering regardless.

– Submit your report on canvas.wpi.edu for grading.