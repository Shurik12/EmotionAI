### Architecture
┌─────────────────┐    ┌──────────────────┐
│   Load Balancer │───▶│  App Server 1    │
│   (Nginx)       │    │  - Instance A    │
└─────────────────┘    └──────────────────┘
         │
         │              ┌──────────────────┐
         └─────────────▶│  App Server 2    │
                        │  - Instance B    │
                        └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │ Dragonfly Cluster│
                        │  - Task Queue    │
                        │  - Shared State  │
                        └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │ Shared Storage   │
                        │  (S3/NFS)        │
                        └──────────────────┘

### Implementation Steps

    Phase 1: Make servers stateless + Dragonfly clustering

    Phase 2: Implement distributed task queue

    Phase 3: Add shared file storage

    Phase 4: Add load balancer + service discovery

    Phase 5: Implement monitoring and auto-scaling