<div align="center" style="background: #ccebd5; border: 1px solid #a9cbb2; border-radius: 18px; padding: 26px 22px; margin: 18px 0; color: #26332a;">
  <h1 style="margin: 0 0 12px; color: #173b29; font-size: 2.2em; font-weight: 800;">RAG Learning</h1>
  <p style="margin: 0; font-size: 1.05em; font-weight: 600;">Current status:
    <span style="display: inline-block; background: #299957; color: #ffffff; border-radius: 999px; padding: 7px 15px; margin-left: 6px; font-weight: 700;">Actively updating the repo</span>
  </p>
</div>

<div style="background: #ded3eb; border: 1px solid #a9a0b2; border-radius: 18px; padding: 22px 24px; margin: 18px 0; color: #292832;">

  <h2 style="margin: 0 0 12px; color: #25232c; font-size: 1.55em; font-weight: 750;">Purpose</h2>
  <p style="margin: 0 0 22px; line-height: 1.7;">A Python project for exploring Retrieval-Augmented Generation (RAG) with LangChain.</p>

  <h2 style="margin: 0 0 12px; color: #25232c; font-size: 1.55em; font-weight: 750;">Goal</h2>
  <ul style="margin: 0 0 22px; padding-left: 22px; line-height: 1.8;">
    <li>Load documents</li>
    <li>Prepare text for retrieval</li>
    <li>Explore embeddings and semantic search</li>
    <li>Generate answers using retrieved context</li>
  </ul>
  <h2 style="margin: 0 0 12px; color: #25232c; font-size: 1.55em; font-weight: 750;">Repo structure</h2>
  <pre style="background: #343a40; border: 1px solid #252a2e; border-radius: 12px; padding: 16px; overflow-x: auto;"><code><span style="color: #f1c75b;">RAG_learning/</span>
├── <span style="color: #75c7ff;">document_loaders/</span>
│   └── <span style="color: #f1f3f5;">text_loader.py</span>
├── <span style="color: #f1f3f5;">main.py</span>
├── <span style="color: #8fe3b0;">pyproject.toml</span>
├── <span style="color: #8fe3b0;">uv.lock</span>
└── <span style="color: #ff9f9f;">README.md</span></code></pre>

  <h2 style="margin: 22px 0 12px; color: #25232c; font-size: 1.55em; font-weight: 750;">Tech stack</h2>
  <p style="margin: 0; line-height: 1.7;">Python · LangChain · Google / Groq integrations · dotenv · Rich</p>

</div>

<div style="background: #d1e2eb; border: 1px solid #a9b7bf; border-radius: 18px; padding: 22px 24px; margin: 18px 0; color: #24232b;">
  <h2 style="margin: 0 0 16px; color: #25232c; font-size: 1.65em; font-weight: 800;">Setup</h2>
  <p style="margin: 0 0 8px; font-weight: 650;">Install dependencies</p>
  <pre style="background: #343a40; border: 1px solid #252a2e; border-radius: 12px; padding: 14px; overflow-x: auto;"><code><span style="color: #f1c75b;">uv</span> <span style="color: #f1f3f5;">sync</span></code></pre>

  <p style="margin: 18px 0 8px; font-weight: 650;">Run the project</p>
  <pre style="background: #343a40; border: 1px solid #252a2e; border-radius: 12px; padding: 14px; overflow-x: auto;"><code><span style="color: #f1c75b;">uv</span> <span style="color: #f1f3f5;">run</span> <span style="color: #f1c75b;">python</span> <span style="color: #f1f3f5;">main.py</span></code></pre>

</div>

<div style="background: #ead8c9; border: 1px solid #b9aa9e; border-radius: 18px; padding: 22px 24px; margin: 18px 0; color: #292832;">

  <h2 style="margin: 0 0 12px; color: #25232c; font-size: 1.55em; font-weight: 750;">Notes</h2>
  <p style="margin: 0; line-height: 1.7;">This repository is a work in progress focused on learning the core steps of a RAG workflow.</p>

</div>
