Security Notes
==============

* 2024-08-25-1: Have enabled snyk vulnerability PR generation

* 2024-08-25-2: Have snyk-ignored CVE-2024-5480. This vulnerability apparently is only relevant when using pytorch.distributed features, which PlantTracerML currently does not use. Ignoring it because the pytorch folks seem disinclined to address it, and refer to a policy never to use the distributed features on an insecure network.
