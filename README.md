# DiffusionModel (WIP)
  - Re-implementing **Denoising Diffusion Probabilistic Models** and **Denoising Diffusion Implicit Models** using Pytorch
  
## Concept of DDPM & DDIM
  - DDPM

  <p align="center">
    <img width="800" alt="concept_of_ddpm" src="https://user-images.githubusercontent.com/22078438/178132101-035afa58-d37b-43ef-ae04-0616a8c044d1.png">
  </p>
  
  - DDIM
  
  <p align="center">
    <img width="800" alt="concept_of_ddim" src="https://user-images.githubusercontent.com/22078438/178132102-202ba6f4-febf-4b52-881c-73d391546f75.png">
   </p>

## Model Architecture (WIP)

## Trining Process (WIP)

## Evaluation Process (WIP)

## Reference
  - Denoising Diffusion Probabilistic Models ([Paper link](https://arxiv.org/pdf/2006.11239.pdf))
  - Denoising Diffusion Implicit Models ([Paper link](https://arxiv.org/pdf/2010.02502.pdf))
  
## Usage of Training
  - DDPM
  ```bash
  python main.py --model_type=ddpm
  ```
  - DDIM
  ```bash
  python main.py --model_type=ddim
  ```
  
## Get a Awesome Generating Images ! (WIP)
  - DDPM
  - DDIM
  
## Experiments
  - Quantitative result

| Model | FID | IS | #Params |
| :---: | :---: | :---: | :---: |
| DDPM | - | - | - |
| DDIM | - | - | - |

  - Qualitative result (WIP, attach more images later !, **Below images are trained model result !**)
    - DDPM
    
    ![model_460000_t_1000_num_0](https://user-images.githubusercontent.com/22078438/178131855-2238f046-a096-4c30-8c5e-29c01cfa6388.png)
    ![model_460000_t_900_num_0](https://user-images.githubusercontent.com/22078438/178131858-226d95f5-81c0-4eeb-be6e-0f1f3f27eb41.png)
    ![model_460000_t_800_num_0](https://user-images.githubusercontent.com/22078438/178131862-d17ad0b5-6b53-4a4a-baf3-1e7494371265.png)
    ![model_460000_t_700_num_0](https://user-images.githubusercontent.com/22078438/178131866-d7f843e2-bed3-4e43-8b30-9e2889065202.png)
    ![model_460000_t_600_num_0](https://user-images.githubusercontent.com/22078438/178131870-8e515d98-ead4-4833-948a-7f5baf771712.png)
    ![model_460000_t_500_num_0](https://user-images.githubusercontent.com/22078438/178131873-d0e3927e-bd78-421d-a7dc-62a9e543e960.png)
    ![model_460000_t_400_num_0](https://user-images.githubusercontent.com/22078438/178131874-619d1890-3eb6-421c-9771-6c119260ee47.png)
    ![model_460000_t_300_num_0](https://user-images.githubusercontent.com/22078438/178131887-09baa8ed-bd4e-4993-919f-f1ca31c30924.png)
    ![model_460000_t_200_num_0](https://user-images.githubusercontent.com/22078438/178131889-67e4bb44-c8ae-404d-9396-1160d183dae5.png)
    ![model_460000_t_100_num_0](https://user-images.githubusercontent.com/22078438/178131901-30563a07-cb85-40fa-a549-cc251fafc0f3.png)
    ![model_460000_t_2_num_0](https://user-images.githubusercontent.com/22078438/178131903-16fa9bd5-f0b4-4c9e-8a13-dc095e47ef8b.png)

    - DDIM
    
    ![model_700000_t_1000_num_0](https://user-images.githubusercontent.com/22078438/178131941-5b697ace-f673-4000-a701-12e1a5dbe4cd.png)
    ![model_700000_t_900_num_0](https://user-images.githubusercontent.com/22078438/178131942-0db4fb4e-2a86-4b2b-a69e-7519e8c14330.png)
    ![model_700000_t_800_num_0](https://user-images.githubusercontent.com/22078438/178131943-517aed85-f9ce-4dc3-b558-517ebaee6fc8.png)
    ![model_700000_t_700_num_0](https://user-images.githubusercontent.com/22078438/178131945-eff91dda-1bf4-49ad-90ce-4844fc1eb252.png)
    ![model_700000_t_600_num_0](https://user-images.githubusercontent.com/22078438/178131947-6d879b84-4091-4d40-996d-a93c2358474e.png)
    ![model_700000_t_500_num_0](https://user-images.githubusercontent.com/22078438/178131949-b0203062-ed0f-42ce-b4ad-cb24419684f5.png)
    ![model_700000_t_400_num_0](https://user-images.githubusercontent.com/22078438/178131951-0bd871d6-9690-4b92-b6a4-22c45e21d920.png)
    ![model_700000_t_300_num_0](https://user-images.githubusercontent.com/22078438/178131952-6aabcb92-ffc4-448f-8869-071e5c8ae5f0.png)
    ![model_700000_t_200_num_0](https://user-images.githubusercontent.com/22078438/178131955-b86db086-f77a-4c0e-beb3-7c2c0b5aba8e.png)
    ![model_700000_t_100_num_0](https://user-images.githubusercontent.com/22078438/178131957-233e5152-20db-4dc5-905b-27122757f616.png)
    ![model_700000_t_2_num_0](https://user-images.githubusercontent.com/22078438/178131961-0ece6783-450f-41dd-b5ec-8c183179d955.png)
    
## Interpolations Experiments (WIP)
  - Example (Below image is paper result)
  
  <img width="828" alt="interpolation_imgs" src="https://user-images.githubusercontent.com/22078438/178132052-9fb621b8-dda6-4298-afaa-ce0437c6cecd.png">

## Metric (FID) Evaluation

  - Usage
  ```bash
  python metric/fid_test.py --cuda=True
  ```
  
## Metric (IS) Evalution (WIP)

  - Usage
  ```bash
  WIP
  ```

--------------------
```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

```bibtex
@article{song2020denoising,
  title={Denoising diffusion implicit models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2010.02502},
  year={2020}
}
```
