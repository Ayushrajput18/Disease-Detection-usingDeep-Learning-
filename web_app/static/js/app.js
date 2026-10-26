document.addEventListener('DOMContentLoaded', () => {
  // Navigation elements
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  const views = {
    'new-analysis': document.getElementById('viewNewAnalysis'),
    'results': document.getElementById('viewResults'),
    'explainability': document.getElementById('viewExplainability'),
    'evaluation': document.getElementById('viewEvaluation'),
    'overview': document.getElementById('viewEvaluation') // default fallback
  };
  const headerBreadcrumb = document.getElementById('headerBreadcrumb');

  // Input Form Elements
  const form = document.getElementById('multimodalForm');
  const btnSubmit = document.getElementById('btnSubmitAnalysis');
  const fileUploadInput = document.getElementById('fileUploadInput');
  const btnReplaceImg = document.getElementById('btnReplaceImg');
  const scopeImage = document.getElementById('scopeImage');
  const dispFileHandle = document.getElementById('dispFileHandle');

  // Clinical Form Fields
  const inputCaseId = document.getElementById('inputCaseId');
  const inputAge = document.getElementById('inputAge');
  const inputSex = document.getElementById('inputSex');
  const inputSite = document.getElementById('inputSite');
  const inputDiameter = document.getElementById('inputDiameter');
  const inputElevation = document.getElementById('inputElevation');
  const inputEvolution = document.getElementById('inputEvolution');
  const inputNotes = document.getElementById('inputNotes');
  const elevationPills = document.querySelectorAll('.pill-toggle-btn');

  // Demo Case Button & Actions
  const btnLoadDemo = document.getElementById('btnLoadDemoCaseDropdown');
  const btnLoadDemoLabel = document.getElementById('btnLoadDemoLabel');
  const btnResetForm = document.getElementById('btnResetForm');
  const btnBackToForm = document.getElementById('btnBackToForm');
  const btnJumpToXAI = document.getElementById('btnJumpToXAI');
  const linkToXAISaliency = document.getElementById('linkToXAISaliency');

  // Results View Elements
  const resPageTitle = document.getElementById('resPageTitle');
  const resStudyId = document.getElementById('resStudyId');
  const resTimestamp = document.getElementById('resTimestamp');
  const resPrimaryName = document.getElementById('resPrimaryName');
  const resPrimaryIcd = document.getElementById('resPrimaryIcd');
  const resPrimaryDesc = document.getElementById('resPrimaryDesc');
  const resPrimaryConfidence = document.getElementById('resPrimaryConfidence');
  const resPrimaryLogit = document.getElementById('resPrimaryLogit');
  const resEpistemicBadge = document.getElementById('resEpistemicBadge');
  const resDiagnosticTableBody = document.getElementById('resDiagnosticTableBody');
  const resVerificationThumb = document.getElementById('resVerificationThumb');

  // Clinical Feature Vector Display in Results
  const dispResAge = document.getElementById('dispResAge');
  const dispResSex = document.getElementById('dispResSex');
  const dispResSite = document.getElementById('dispResSite');
  const dispResDiameter = document.getElementById('dispResDiameter');
  const dispResElevation = document.getElementById('dispResElevation');
  const dispResEvolution = document.getElementById('dispResEvolution');

  // Sub-Model Ablation Displays
  const dispAblImgClass = document.getElementById('dispAblImgClass');
  const dispAblImgConf = document.getElementById('dispAblImgConf');
  const dispAblMetaClass = document.getElementById('dispAblMetaClass');
  const dispAblMetaConf = document.getElementById('dispAblMetaConf');
  const dispAblFusionClass = document.getElementById('dispAblFusionClass');
  const dispAblFusionConf = document.getElementById('dispAblFusionConf');
  const dispAblFusionSigma = document.getElementById('dispAblFusionSigma');

  const barImgConf = document.getElementById('barImgConf');
  const textImgConf = document.getElementById('textImgConf');
  const barMetaConf = document.getElementById('barMetaConf');
  const textMetaConf = document.getElementById('textMetaConf');
  const barFusionConf = document.getElementById('barFusionConf');
  const textFusionConf = document.getElementById('textFusionConf');

  // Explainability Lab Elements
  const xaiCaseSelector = document.getElementById('xaiCaseSelector');
  const xaiTargetClassPill = document.getElementById('xaiTargetClassPill');
  const xaiOriginalImg = document.getElementById('xaiOriginalImg');
  const xaiGradcamImg = document.getElementById('xaiGradcamImg');
  const dispPeakSaliency = document.getElementById('dispPeakSaliency');
  const dispCentroid = document.getElementById('dispCentroid');
  const dispPathologyFeature = document.getElementById('dispPathologyFeature');
  const dispPerimeter = document.getElementById('dispPerimeter');
  const sliderOpacity = document.getElementById('sliderOpacity');
  const textOpacityVal = document.getElementById('textOpacityVal');
  const shapBarsContainer = document.getElementById('shapBarsContainer');

  // App State
  let allDemoCases = [];
  let currentCaseId = 'STUDY-2024-DM-0941';
  let uploadedFile = null;
  let lastAnalysisData = null;

  // 1. Navigation Controller
  function switchView(viewKey) {
    Object.keys(views).forEach(key => {
      if (views[key]) {
        views[key].style.display = (key === viewKey) ? 'block' : 'none';
      }
    });

    navItems.forEach(item => {
      if (item.dataset.view === viewKey) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    const breadcrumbs = {
      'new-analysis': 'Clinical Protocol Engine › New Analysis',
      'results': 'Clinical Protocol Engine › Multimodal Analysis Results',
      'explainability': 'Clinical Protocol Engine › Explainability & Interpretability Lab',
      'evaluation': 'Clinical Protocol Engine › Benchmark Evaluation',
      'overview': 'Clinical Protocol Engine › Overview'
    };
    headerBreadcrumb.textContent = breadcrumbs[viewKey] || 'Clinical Protocol Engine';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      switchView(item.dataset.view);
    });
  });

  if (btnBackToForm) {
    btnBackToForm.addEventListener('click', () => switchView('new-analysis'));
  }

  if (btnJumpToXAI) {
    btnJumpToXAI.addEventListener('click', () => switchView('explainability'));
  }

  if (linkToXAISaliency) {
    linkToXAISaliency.addEventListener('click', (e) => {
      e.preventDefault();
      switchView('explainability');
    });
  }

  // 2. Fetch Initial Demo Cases from Backend
  fetch('/api/cases')
    .then(r => r.json())
    .then(data => {
      if (data.cases) {
        allDemoCases = data.cases;
        loadCase(allDemoCases[0]);
      }
    })
    .catch(err => console.error('Error fetching demo cohorts:', err));

  // 3. Load Case Function
  function loadCase(c) {
    currentCaseId = c.id;
    uploadedFile = null;

    inputCaseId.value = c.id;
    inputAge.value = c.age;
    inputSex.value = c.sex;
    inputSite.value = c.site;
    inputDiameter.value = c.diameter || 8.5;
    inputElevation.value = c.elevation || 'Plaque / Palpable';
    inputEvolution.value = c.evolution || 'Enlarging lesion (>3 months duration)';
    inputNotes.value = c.notes || '';

    // Update elevation buttons
    elevationPills.forEach(p => {
      if (p.dataset.val === c.elevation) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });

    // Update microscope view
    scopeImage.src = c.thumb;
    dispFileHandle.textContent = c.filename.substring(0, 16) + '...';

    // Update demo dropdown label
    btnLoadDemoLabel.textContent = `Demo: ${c.id} (${c.diagnosis})`;

    // Sync selector in Explainability Lab
    if (xaiCaseSelector) {
      xaiCaseSelector.value = c.id;
    }
  }

  // 4. Elevation pill toggles
  elevationPills.forEach(pill => {
    pill.addEventListener('click', () => {
      elevationPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      inputElevation.value = pill.dataset.val;
    });
  });

  // 5. Demo case cycle on click
  let demoIdx = 0;
  btnLoadDemo.addEventListener('click', () => {
    if (allDemoCases.length === 0) return;
    demoIdx = (demoIdx + 1) % allDemoCases.length;
    loadCase(allDemoCases[demoIdx]);
  });

  // 6. Reset form
  btnResetForm.addEventListener('click', () => {
    if (allDemoCases.length > 0) {
      loadCase(allDemoCases[0]);
    }
  });

  // 7. Replace / Upload image
  btnReplaceImg.addEventListener('click', () => fileUploadInput.click());
  fileUploadInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      uploadedFile = file;
      currentCaseId = null;

      const reader = new FileReader();
      reader.onload = (ev) => {
        scopeImage.src = ev.target.result;
        dispFileHandle.textContent = file.name.substring(0, 16) + '...';
      };
      reader.readAsDataURL(file);
    }
  });

  // 8. Opacity Slider for Grad-CAM
  sliderOpacity.addEventListener('input', (e) => {
    const val = e.target.value;
    textOpacityVal.textContent = `${val}%`;
    xaiGradcamImg.style.opacity = (val / 100);
  });

  // 9. Case selector change in Explainability Lab
  if (xaiCaseSelector) {
    xaiCaseSelector.addEventListener('change', (e) => {
      const selected = allDemoCases.find(c => c.id === e.target.value);
      if (selected) {
        loadCase(selected);
        triggerAnalysis(false); // Run silently to sync XAI
      }
    });
  }

  // 10. Form Submission (Run Analysis)
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    triggerAnalysis(true);
  });

  function triggerAnalysis(navigateToResults = true) {
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = '<span>⟳ Processing Ingestion Pipeline...</span>';

    const formData = new FormData();
    if (uploadedFile) {
      formData.append('image', uploadedFile);
    } else {
      formData.append('case_id', currentCaseId);
    }

    formData.append('age', inputAge.value);
    formData.append('sex', inputSex.value);
    formData.append('site', inputSite.value);
    formData.append('diameter', inputDiameter.value);
    formData.append('elevation', inputElevation.value);
    formData.append('evolution', inputEvolution.value);
    formData.append('notes', inputNotes.value);

    fetch('/api/analyze', {
      method: 'POST',
      body: formData
    })
      .then(r => r.json())
      .then(data => {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Multimodal Analysis</span>';

        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        lastAnalysisData = data;
        populateResultsView(data);
        populateXAIView(data);

        if (navigateToResults) {
          switchView('results');
        }
      })
      .catch(err => {
        console.error(err);
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = '<span>Run Multimodal Analysis</span>';
        alert('Analysis connection error: ' + err.message);
      });
  }

  // 11. Populate Results View
  function populateResultsView(data) {
    const primary = data.primary;
    resPageTitle.textContent = `Analysis Results: ${data.session_id}`;
    resStudyId.textContent = data.session_id;
    resTimestamp.textContent = data.timestamp;

    resPrimaryName.textContent = primary.name;
    resPrimaryIcd.textContent = primary.icd_code;
    resPrimaryDesc.textContent = primary.description;
    resPrimaryConfidence.textContent = `${primary.confidence_pct}%`;
    resPrimaryLogit.textContent = `Logit: ${primary.raw_logit > 0 ? '+' : ''}${primary.raw_logit}`;
    resEpistemicBadge.textContent = `● σ = ${primary.sigma_uncertainty} (${primary.uncertainty_label})`;

    // Verification Image
    resVerificationThumb.src = data.image_b64;

    // Feature Vector
    dispResAge.textContent = `${data.clinical_features.age} years old`;
    dispResSex.textContent = data.clinical_features.sex;
    dispResSite.textContent = data.clinical_features.site;
    dispResDiameter.textContent = `${data.clinical_features.diameter} mm`;
    dispResElevation.textContent = data.clinical_features.elevation;
    dispResEvolution.textContent = data.clinical_features.evolution;

    // Diagnostic table rows
    resDiagnosticTableBody.innerHTML = '';
    data.diagnostic_table.forEach(row => {
      const tr = document.createElement('tr');
      if (row.is_predicted) tr.className = 'active-predicted-row';

      tr.innerHTML = `
        <td class="class-rank-col">${row.rank}</td>
        <td class="class-name-col">
          <strong>${row.name}</strong> (${row.code})
          ${row.is_predicted ? '<span class="badge-predicted-tag">Predicted</span>' : ''}
        </td>
        <td class="class-logit-col">${row.raw_logit > 0 ? '+' : ''}${row.raw_logit}</td>
        <td class="class-conf-col">
          <div class="conf-bar-wrapper">
            <div class="conf-mini-track">
              <div class="conf-mini-fill" style="width: ${Math.max(row.confidence_pct, 1)}%"></div>
            </div>
            <span class="conf-pct-text">${row.confidence_pct}%</span>
          </div>
        </td>
      `;
      resDiagnosticTableBody.appendChild(tr);
    });

    // Sub-Model Ablation
    const sub = data.submodels;
    dispAblImgClass.textContent = sub.image_only.name;
    dispAblImgConf.textContent = `${sub.image_only.confidence_pct}% (${(sub.image_only.confidence_pct/100).toFixed(3)})`;

    dispAblMetaClass.textContent = sub.clinical_only.name;
    dispAblMetaConf.textContent = `${sub.clinical_only.confidence_pct}% (${(sub.clinical_only.confidence_pct/100).toFixed(3)})`;

    dispAblFusionClass.textContent = sub.fused_multimodal.name;
    dispAblFusionConf.textContent = `${sub.fused_multimodal.confidence_pct}% (+${sub.fused_multimodal.synergistic_gain}% gain)`;
    dispAblFusionSigma.textContent = `σ = ${sub.fused_multimodal.sigma} (-50% var)`;

    // Lift bars
    barImgConf.style.width = `${sub.image_only.confidence_pct}%`;
    textImgConf.textContent = `${sub.image_only.confidence_pct}%`;

    barMetaConf.style.width = `${sub.clinical_only.confidence_pct}%`;
    textMetaConf.textContent = `${sub.clinical_only.confidence_pct}%`;

    barFusionConf.style.width = `${sub.fused_multimodal.confidence_pct}%`;
    textFusionConf.textContent = `${sub.fused_multimodal.confidence_pct}%`;
  }

  // 12. Populate Explainability Lab View
  function populateXAIView(data) {
    xaiTargetClassPill.textContent = `● Target: ${data.primary.short_name} (${data.primary.code}) p=${(data.primary.confidence_pct/100).toFixed(3)}`;
    xaiOriginalImg.src = data.image_b64;

    if (data.gradcam && data.gradcam.overlay_b64) {
      xaiGradcamImg.src = data.gradcam.overlay_b64;
      dispPeakSaliency.textContent = data.gradcam.peak_saliency;
      dispCentroid.textContent = `[x: ${data.gradcam.centroid[0]} px, y: ${data.gradcam.centroid[1]} px]`;
    }

    dispPathologyFeature.textContent = data.primary.pathology_features;
    dispPerimeter.textContent = data.primary.perimeter_desc;

    // Render SHAP waterfall bars
    shapBarsContainer.innerHTML = '';
    data.shap_features.forEach(feat => {
      const row = document.createElement('div');
      row.className = 'shap-item-row';
      row.innerHTML = `
        <span class="shap-feature-name">${feat.name}</span>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${feat.type}" style="width: ${feat.pct}%"></div>
        </div>
        <span class="shap-val-text ${feat.type}">${feat.value}</span>
      `;
      shapBarsContainer.appendChild(row);
    });
  }

  // Run initial analysis silently to populate results
  setTimeout(() => {
    triggerAnalysis(false);
  }, 300);
});
