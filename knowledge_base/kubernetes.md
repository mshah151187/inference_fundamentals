# Kubernetes (K8s) — From Scratch

## Why Kubernetes Exists

You have a model serving process. It runs on one machine. Traffic doubles — you need two machines. One machine crashes — the process dies. You deploy a new version — old requests drop.

Kubernetes solves: **how do you run many processes reliably across many machines?**

Before K8s, you SSHed into boxes, started processes manually, wrote bash scripts to restart them. K8s makes the cluster look like one computer and handles process lifecycle, placement, scaling, and networking automatically.

---

## Mental Model

```
You say:  "I want 3 copies of my inference server running at all times"
K8s does: figures out which machines have capacity, places the containers,
          restarts them if they crash, replaces them if the machine dies
```

You declare **what you want** (desired state). K8s continuously reconciles **what exists** (actual state) toward your desired state. This is the **control loop** pattern — the fundamental idea behind everything in K8s.

---

## Cluster Anatomy

```
┌─────────────────────────────────────────────────────────┐
│                        CLUSTER                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │               CONTROL PLANE                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  │  API     │  │  etcd    │  │  Scheduler    │  │   │
│  │  │  Server  │  │  (state) │  │  Controller   │  │   │
│  │  └──────────┘  └──────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Node 1  │  │  Node 2  │  │  Node 3  │  ← Workers   │
│  │ [Pod][Pod│  │ [Pod]    │  │ [Pod][Pod│              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## Core Building Blocks

### Pod

The smallest deployable unit. A pod is one or more containers that:
- Share the same network namespace (same IP, same localhost)
- Share the same storage volumes
- Are always scheduled together on the same node

```yaml
# A pod running one inference server container
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: inference-server
      image: myrepo/gpt2-server:v1
      resources:
        limits:
          nvidia.com/gpu: 1
```

**Key point**: You almost never create Pods directly. You create higher-level objects (Deployment, Job) that create Pods for you. If a Pod dies, K8s doesn't restart it — but a Deployment will create a new one.

---

### Node

A physical or virtual machine in the cluster. Each node runs:
- **kubelet**: agent that talks to the control plane, starts/stops containers
- **kube-proxy**: handles networking rules
- **container runtime**: Docker/containerd that actually runs containers

Nodes have resources (CPU, RAM, GPU). The Scheduler places pods on nodes based on what resources the pod requests.

---

### Control Plane Components

| Component | Role |
|-----------|------|
| **API Server** | Single entry point for all K8s operations. `kubectl` talks to this. |
| **etcd** | Distributed key-value store. The only source of truth for cluster state. |
| **Scheduler** | Watches for new Pods with no node assigned, picks the best node. |
| **Controller Manager** | Runs control loops — ReplicaSet controller, Deployment controller, etc. Each loop watches state and reconciles toward desired. |

**Control loop pattern** (how every controller works):
```
while True:
    desired = read_desired_state_from_etcd()
    actual  = observe_actual_state()
    if actual != desired:
        take_action_to_reconcile()
```

---

### Deployment

Declares: "I want N replicas of this pod, with this rolling-update strategy."

The Deployment controller creates a **ReplicaSet**, which creates and maintains N Pods.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
spec:
  replicas: 3                    # desired state: 3 pods
  selector:
    matchLabels:
      app: inference-server
  template:
    spec:
      containers:
        - name: server
          image: myrepo/gpt2-server:v1
          resources:
            limits:
              nvidia.com/gpu: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1          # at most 1 pod down during update
      maxSurge: 1                # at most 1 extra pod during update
```

**Rolling update**: when you push a new image tag, K8s replaces pods one by one — keeps N-1 running at all times. Zero-downtime deploys by default.

---

### Service

Pods get random IPs that change when they're replaced. A **Service** gives you a stable virtual IP (ClusterIP) that load-balances across all pods matching a label selector.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-svc
spec:
  selector:
    app: inference-server         # routes to all pods with this label
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP                 # only reachable inside the cluster
```

Types:
- **ClusterIP**: internal-only stable IP
- **NodePort**: exposes on every node's IP at a fixed port (for dev/testing)
- **LoadBalancer**: provisions a cloud load balancer (AWS ELB, GCP LB) — used in production to expose externally

---

### Ingress

Layer 7 (HTTP) routing. Routes external HTTP traffic to internal Services based on path or host rules.

```
client → Ingress → /api/infer → inference-svc → pods
                 → /api/health → health-svc   → pods
```

Requires an **Ingress Controller** running in the cluster (nginx-ingress, Traefik, etc.).

---

### ConfigMap & Secret

Decouple configuration from container images.

```yaml
# ConfigMap: non-sensitive config
kind: ConfigMap
data:
  MODEL_NAME: "gpt2"
  MAX_BATCH_SIZE: "64"

# Secret: sensitive data (base64 encoded)
kind: Secret
data:
  HF_TOKEN: <base64>
```

Injected into pods as environment variables or mounted as files.

---

### Namespace

Virtual cluster within a cluster. Namespaces partition resources — useful for isolating teams, environments (dev/staging/prod), or projects.

```
cluster
├── namespace: production    (inference-server, model-store)
├── namespace: staging       (inference-server, model-store)
└── namespace: monitoring    (prometheus, grafana)
```

---

### PersistentVolume (PV) and PersistentVolumeClaim (PVC)

Pods are ephemeral — their local storage dies with them. PVs are cluster-level storage resources (backed by NFS, EBS, GCS, etc.). A PVC is a pod's request for storage.

```yaml
kind: PersistentVolumeClaim
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Gi              # I need 100GB
```

K8s binds the PVC to an available PV. The pod mounts it. Data persists across pod restarts.

---

## GPU in Kubernetes

K8s doesn't know about GPUs natively. You need the **NVIDIA Device Plugin** — a DaemonSet that runs on every GPU node, advertises `nvidia.com/gpu` as a resource, and ensures containers get exclusive access to physical GPUs.

```yaml
# Request a GPU in a pod spec
resources:
  limits:
    nvidia.com/gpu: 1            # exclusive access to 1 GPU
```

GPU scheduling is **all-or-nothing** per GPU — K8s can't split a GPU between pods (unless you use MIG or MPS at the driver level separately).

---

## Autoscaling

### Horizontal Pod Autoscaler (HPA)
Scales the number of pod replicas based on CPU/memory metrics (or custom metrics from Prometheus).

```
observed avg CPU > 70%  →  scale up pods
observed avg CPU < 30%  →  scale down pods
```

### Vertical Pod Autoscaler (VPA)
Adjusts the CPU/memory resource requests of existing pods (restarts them with new limits).

### Cluster Autoscaler
Adds or removes **nodes** when pods can't be scheduled (node full) or nodes are underutilized.

```
HPA adds pods → pods can't fit on existing nodes → Cluster Autoscaler adds a node
```

---

## Jobs and CronJobs

For batch workloads (not long-running servers):

```yaml
kind: Job
spec:
  completions: 1                 # run to completion once
  parallelism: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: myrepo/train:v1
```

A **CronJob** runs a Job on a schedule (like cron). Used for periodic batch inference, data pipeline runs, etc.

---

## DaemonSet

Ensures exactly one pod runs on every node (or every node matching a selector). Used for:
- Node-level monitoring (Prometheus node exporter)
- Log collectors (Fluentd)
- GPU device plugins (NVIDIA device plugin)

---

## Resource Requests vs Limits

```yaml
resources:
  requests:
    cpu: "2"          # scheduler uses this to decide where to place the pod
    memory: "8Gi"
  limits:
    cpu: "4"          # hard cap — process throttled if exceeded
    memory: "16Gi"    # hard cap — process OOM-killed if exceeded
```

**Request** = guaranteed minimum. **Limit** = hard ceiling. Pod is scheduled based on requests, throttled/killed based on limits. Always set both for production workloads.

---

## KubeRay — Ray on Kubernetes

### What Ray Is

Ray is a Python framework for distributed computing. It lets you:
- Run Python functions across many machines transparently
- Build distributed serving systems (`ray.serve`)
- Coordinate distributed training (`ray.train`)
- Manage worker pools for batch inference

### What KubeRay Is

KubeRay is a Kubernetes operator that manages Ray clusters on K8s. An **operator** is a pattern: you install a custom controller + custom resource definitions (CRDs), and the controller manages domain-specific objects (Ray clusters) using the K8s control loop pattern.

```
kubectl apply -f ray-cluster.yaml
         ↓
KubeRay operator sees RayCluster CRD
         ↓
Creates: 1 head pod + N worker pods + Services
         ↓
Ray cluster is ready — workers register with head
```

### RayCluster CRD

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: inference-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-gpu
            resources:
              limits:
                cpu: 4
                memory: 16Gi

  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      minReplicas: 1
      maxReplicas: 8                  # autoscaling
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:2.9.0-gpu
              resources:
                limits:
                  nvidia.com/gpu: 1
                  cpu: 8
                  memory: 32Gi
```

### RayCluster Topology

```
┌──────────────────────────────────────────────────┐
│                  RayCluster                      │
│                                                  │
│  ┌─────────────┐      ┌──────────┐ ┌──────────┐  │
│  │  Head Pod   │      │ Worker 1 │ │ Worker 2 │  │
│  │             │◄────►│ (1 GPU)  │ │ (1 GPU)  │  │
│  │ - GCS       │      └──────────┘ └──────────┘  │
│  │ - Dashboard │      ┌──────────┐ ┌──────────┐  │
│  │ - Raylet    │◄────►│ Worker 3 │ │ Worker 4 │  │
│  └─────────────┘      │ (1 GPU)  │ │ (1 GPU)  │  │
│                       └──────────┘ └──────────┘  │
└──────────────────────────────────────────────────┘
```

**Head node**: runs the Global Control Store (GCS) — tracks all actors, tasks, object references. One head per cluster.

**Worker nodes**: run tasks and actors. Each worker has a Raylet (local scheduler) that interfaces with the head.

### RayService — Serving with KubeRay

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serveConfigV2: |
    applications:
      - name: llm
        route_prefix: /
        import_path: serve_app:deployment
        deployments:
          - name: LLMDeployment
            num_replicas: 2
            ray_actor_options:
              num_gpus: 1
  rayClusterConfig:
    # ... same as RayCluster above
```

RayService adds: zero-downtime updates (waits for new version to pass health checks before cutting over), auto-recovery, HTTP routing via a K8s Service.

### RayJob — Batch Jobs with KubeRay

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: batch-inference
spec:
  entrypoint: python batch_infer.py
  shutdownAfterJobFinishes: true     # cluster torn down after job completes
  rayClusterSpec:
    # ... cluster config
```

**Key difference from RayService**: RayJob creates a cluster, runs a job to completion, then deletes the cluster. Cost-efficient for batch workloads.

## Helm — Package Manager for Kubernetes

**The problem**: Installing Prometheus on K8s requires 15+ YAML files — Deployments, Services, ConfigMaps, RBAC roles, CRDs. You don't want to write and maintain all of that yourself.

**What Helm does**: Someone packages all those YAMLs into a **chart**. You install it with one command, passing only the config you care about.

```bash
# Without Helm: download 15 YAMLs, edit each, kubectl apply each
# With Helm:
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --set grafana.adminPassword=secret \
  --set prometheus.retention=30d
```

### Three Concepts

| Concept | What it is |
|---------|-----------|
| **Chart** | The package — a directory of templated YAML files + `values.yaml` with all defaults. Lives in a Helm repository (like a package registry). |
| **Release** | One installed instance of a chart. Install the same chart twice with different names → two independent releases (e.g., prod-prometheus and staging-prometheus). |
| **Values** | Config you pass to override chart defaults. Via `--set` flags or your own `values.yaml` file. |

### Typical values.yaml (app team owns this)

```yaml
# values.yaml — override just what you need
replicaCount: 4
image:
  repository: myrepo/vllm-server
  tag: v1.3
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: 32Gi
service:
  type: ClusterIP
  port: 80
ingress:
  enabled: true
  host: inference.mycompany.com
```

```bash
helm install inference-server ./my-inference-chart -f values.yaml -n inference-prod
helm upgrade inference-server ./my-inference-chart -f values.yaml -n inference-prod  # update
helm rollback inference-server 1 -n inference-prod                                   # roll back to revision 1
```

### Who uses Helm

- **Platform team**: installs third-party tools (GPU Operator, Prometheus, nginx-ingress, KubeRay operator) using community charts from registries like Artifact Hub
- **App team**: may package their own Deployment/Service/Ingress/HPA into a chart so deployments across environments (dev/staging/prod) are a single `helm upgrade` with different `values.yaml` per env

---

## Key Patterns for ML Inference on K8s

### Pattern 1: Deployment + Service (simple stateless serving)
```
Deployment (N replicas, 1 GPU each) → Service (ClusterIP) → Ingress → clients
```
Best for: homogeneous model serving, easy horizontal scaling.

### Pattern 2: RayService (model parallelism, dynamic batching)
```
RayService → RayCluster (head + GPU workers) → Ray Serve handles routing + batching
```
Best for: large models needing tensor parallelism, variable batch sizes, multiple model versions.

### Pattern 3: RayJob (batch inference)
```
RayJob → ephemeral RayCluster → process dataset → cluster torn down
```
Best for: offline inference over large datasets, training runs.

---

## Key kubectl Commands

```bash
# Cluster state
kubectl get nodes                          # list nodes and status
kubectl describe node <name>               # node resources, GPU capacity
kubectl top nodes                          # live CPU/mem usage

# Workloads
kubectl get pods -n <namespace>            # list pods
kubectl describe pod <name>                # events, resource usage, errors
kubectl logs <pod-name> -c <container>     # container logs
kubectl exec -it <pod> -- bash             # shell into running pod

# Apply / delete
kubectl apply -f manifest.yaml             # create or update resources
kubectl delete -f manifest.yaml            # delete resources
kubectl rollout status deployment/<name>   # watch rolling update progress
kubectl rollout undo deployment/<name>     # rollback to previous version

# Resource usage
kubectl top pods --containers              # CPU/mem per container
kubectl get events --sort-by=.lastTimestamp  # recent cluster events
```

---

## Interview Vocabulary Quick Reference

| Term | One-line meaning |
|------|-----------------|
| **Pod** | Smallest unit; one or more co-located containers |
| **Deployment** | Declares desired replica count; handles rolling updates |
| **ReplicaSet** | Ensures N pods running; created by Deployment |
| **Service** | Stable virtual IP load-balancing across pod replicas |
| **Ingress** | HTTP routing rules (path/host → Service) |
| **Node** | Physical/VM machine; runs kubelet + pods |
| **Control plane** | API server + etcd + scheduler + controllers |
| **etcd** | The only source of truth — all cluster state lives here |
| **Scheduler** | Places pods on nodes based on resource requests |
| **Controller** | Control loop: watches state, reconciles toward desired |
| **Operator** | Custom controller + CRDs for domain-specific objects (e.g., Ray) |
| **CRD** | Custom Resource Definition — extend K8s API with new object types |
| **DaemonSet** | One pod per node (monitoring, GPU plugin) |
| **StatefulSet** | Like Deployment but with stable identity (for databases) |
| **ConfigMap/Secret** | Config injection decoupled from container image |
| **PVC/PV** | Persistent storage claim / provisioned storage |
| **HPA** | Horizontal Pod Autoscaler — scales replica count |
| **Cluster Autoscaler** | Adds/removes nodes based on pending pods |
| **Namespace** | Virtual cluster partition |
| **KubeRay** | Kubernetes operator for managing Ray clusters |
| **RayCluster** | CRD: head + worker pods forming a Ray cluster |
| **RayService** | CRD: long-running Ray Serve deployment with zero-downtime updates |
| **RayJob** | CRD: ephemeral cluster for one-off batch jobs |
| **Resource request** | What scheduler uses to place pod (guaranteed minimum) |
| **Resource limit** | Hard cap; OOM kill on memory breach, throttle on CPU |

---

## Production Playbook: Blank GPU Nodes → Scalable Inference Service

This walks through what actually happens in order, who does what, and why.

---

### Phase 0 — Hardware and Cloud (Infra / Cloud Team)

Before K8s exists at all, someone provisions the machines.

**What happens:**
- Request or purchase GPU nodes (e.g., 8x H100 nodes on AWS as `p4d.24xlarge`, or bare-metal)
- Set up networking: nodes get IPs, subnets, firewall rules that allow pod-to-pod traffic
- Attach block storage volumes (for model weights, checkpoints)

**Who does it:** Cloud/infra team or physical data center team. Application team never touches this.

---

### Phase 1 — Cluster Bootstrap (Platform Team)

Start with blank nodes. Install Kubernetes itself.

**Step 1a: Install K8s control plane and join nodes**

Options:
- **Managed**: AWS EKS / GCP GKE / Azure AKS — cloud provisions control plane for you, you just add worker nodes
- **Self-managed**: `kubeadm init` on a dedicated control-plane node, then `kubeadm join` on each worker

After this step:
```bash
kubectl get nodes
# NAME          STATUS     ROLES           AGE
# node-gpu-01   NotReady   <none>          1m    ← not ready yet, no network
# node-gpu-02   NotReady   <none>          1m
```

**Step 1b: Install CNI (Container Network Interface)**

K8s doesn't include networking — you plug in a CNI plugin. It makes pods on different nodes able to talk to each other with real IPs.

Common choices: Cilium (most modern, eBPF-based), Calico, Flannel.

```bash
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
# Now nodes flip to Ready
```

**Step 1c: Install NVIDIA GPU Operator**

A single operator that installs everything GPU-related on every GPU node automatically:
- NVIDIA drivers (if not pre-installed)
- Container Toolkit (so containers can access GPUs)
- Device Plugin (advertises `nvidia.com/gpu` as schedulable resource)
- DCGM exporter (GPU metrics for Prometheus)

```bash
helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace
```

After this:
```bash
kubectl describe node node-gpu-01 | grep nvidia
# Capacity:
#   nvidia.com/gpu: 8          ← K8s now knows this node has 8 GPUs
```

**Step 1d: Install storage provisioner**

So pods can claim persistent volumes (for model weights, logs).

```bash
# Example: AWS EBS CSI driver for EBS-backed PVCs
helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver
```

**Step 1e: Install monitoring stack**

```bash
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack -n monitoring
# Installs: Prometheus + Grafana + node-exporter + kube-state-metrics + DCGM dashboards
```

**Step 1f: Install Ingress Controller**

```bash
helm install ingress-nginx ingress-nginx/ingress-nginx
# Creates a LoadBalancer Service → cloud provisions an external IP
```

After Phase 1, the cluster is ready. The platform team hands over to application teams.

**Platform team deliverable**: a kubeconfig file granting the application team access to their namespace.

---

### Phase 2 — Cluster Config for Application Teams (Platform Team)

Platform team sets up guard rails so application teams can't interfere with each other or consume unbounded resources.

**Step 2a: Create namespace**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inference-prod
  labels:
    team: ml-platform
```

**Step 2b: Set ResourceQuota — cap what the namespace can consume**

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: inference-quota
  namespace: inference-prod
spec:
  hard:
    requests.cpu: "64"
    requests.memory: 512Gi
    requests.nvidia.com/gpu: "16"    # max 16 GPUs for this namespace
    limits.nvidia.com/gpu: "16"
    pods: "50"
```

**Step 2c: Set LimitRange — default limits when app team forgets to set them**

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: inference-prod
spec:
  limits:
    - type: Container
      default:
        cpu: "2"
        memory: 8Gi
      defaultRequest:
        cpu: "1"
        memory: 4Gi
```

**Step 2d: Taint GPU nodes — prevent non-GPU pods from landing on GPU machines**

```bash
kubectl taint nodes node-gpu-01 gpu=true:NoSchedule
kubectl taint nodes node-gpu-02 gpu=true:NoSchedule
# Only pods with a matching toleration can be scheduled on GPU nodes
```

**Step 2e: Label nodes — so pods can target specific GPU types**

```bash
kubectl label nodes node-gpu-01 accelerator=h100
kubectl label nodes node-gpu-02 accelerator=h100
```

**Step 2f: RBAC — grant application team access**

```yaml
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  namespace: inference-prod
  name: app-team-role
rules:
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["get", "list", "create", "update", "patch"]
  # NOT allowed: nodes, namespaces, resourcequotas — platform team only
```

**Platform team sets. Application team cannot change any of Phase 2.**

---

### Phase 3 — Application Team: Write Manifests

Application team owns everything in their namespace. They write YAML manifests checked into git (GitOps pattern — ArgoCD or Flux watches the repo and applies changes automatically).

**Step 3a: Store model weights in a PVC**

```yaml
# pvc.yaml — claim 500GB for model weights
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights-pvc
  namespace: inference-prod
spec:
  accessModes: [ReadWriteMany]       # multiple pods can read simultaneously
  resources:
    requests:
      storage: 500Gi
  storageClassName: efs-sc           # EFS so all GPU nodes can mount it
```

Model weights are downloaded once, written to this PVC. All inference pods mount it read-only.

**Step 3b: ConfigMap for model config**

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
  namespace: inference-prod
data:
  MODEL_NAME: "meta-llama/Llama-3-8B"
  MAX_BATCH_SIZE: "32"
  MAX_SEQ_LEN: "4096"
  TENSOR_PARALLEL_SIZE: "1"
```

**Step 3c: Secret for API keys**

```bash
kubectl create secret generic inference-secrets \
  --from-literal=HF_TOKEN=hf_xxx \
  -n inference-prod
```

**Step 3d: Deployment**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  namespace: inference-prod
spec:
  replicas: 4                          # 4 pods, 1 GPU each = 4 GPUs total
  selector:
    matchLabels:
      app: inference-server
  template:
    metadata:
      labels:
        app: inference-server
    spec:
      # Target GPU nodes (matches platform team's label + tolerate their taint)
      nodeSelector:
        accelerator: h100
      tolerations:
        - key: "gpu"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"

      # Wait for model weights to be present before starting
      initContainers:
        - name: check-weights
          image: busybox
          command: ["sh", "-c", "until [ -f /models/model.safetensors ]; do sleep 5; done"]
          volumeMounts:
            - name: model-weights
              mountPath: /models

      containers:
        - name: inference-server
          image: myrepo/vllm-server:v1.2
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: inference-config
            - secretRef:
                name: inference-secrets
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              nvidia.com/gpu: "1"     # hard limit: exactly 1 GPU per pod
          volumeMounts:
            - name: model-weights
              mountPath: /models
              readOnly: true

          # Readiness probe: only send traffic when model is loaded
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60    # model load takes time
            periodSeconds: 10

          # Liveness probe: restart if server hangs
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
            failureThreshold: 3

      volumes:
        - name: model-weights
          persistentVolumeClaim:
            claimName: model-weights-pvc
```

**Step 3e: Service**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-svc
  namespace: inference-prod
spec:
  selector:
    app: inference-server
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

**Step 3f: Ingress — route external traffic in**

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
  namespace: inference-prod
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"    # long inference timeout
spec:
  rules:
    - host: inference.mycompany.com
      http:
        paths:
          - path: /v1/generate
            pathType: Prefix
            backend:
              service:
                name: inference-svc
                port:
                  number: 80
```

**Step 3g: Horizontal Pod Autoscaler**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: inference-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 16                       # capped by namespace quota (16 GPUs)
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Or custom metric: requests per second from Prometheus
    - type: Pods
      pods:
        metric:
          name: inference_requests_per_second
        target:
          type: AverageValue
          averageValue: "20"            # scale up when > 20 RPS per pod
```

---

### Phase 4 — Deploy

```bash
# Apply everything
kubectl apply -f pvc.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Watch rollout
kubectl rollout status deployment/inference-server -n inference-prod

# Check pods landed on GPU nodes
kubectl get pods -n inference-prod -o wide
# NAME                         NODE          STATUS
# inference-server-7d9f-xkp2   node-gpu-01   Running
# inference-server-7d9f-mn8q   node-gpu-02   Running
```

---

### Phase 5 — Update (Zero-Downtime)

Application team pushes a new image:

```bash
kubectl set image deployment/inference-server \
  inference-server=myrepo/vllm-server:v1.3 \
  -n inference-prod
```

K8s does a rolling update:
1. Spins up 1 new pod with v1.3
2. Waits for readiness probe to pass (model loaded)
3. Removes 1 old v1.2 pod
4. Repeats until all pods are v1.3

At no point are all pods down simultaneously. Old pods drain in-flight requests before shutdown (configurable via `terminationGracePeriodSeconds`).

---

### Responsibility Summary

| Task | Platform Team | App Team |
|------|--------------|----------|
| Provision cloud nodes | ✓ | |
| Install K8s, CNI, GPU operator | ✓ | |
| Install Prometheus, Ingress controller | ✓ | |
| Create namespaces, ResourceQuota, RBAC | ✓ | |
| Taint/label GPU nodes | ✓ | |
| Write Deployment, Service, Ingress | | ✓ |
| Set resource requests/limits per pod | | ✓ |
| Write HPA rules | | ✓ |
| Manage ConfigMaps, Secrets | | ✓ |
| Deploy and roll back | | ✓ |
| Monitor app-level metrics (latency, errors) | | ✓ |
| Monitor cluster-level metrics (node health, GPU util) | ✓ | |

---

### Request Flow End-to-End

```
Client (curl / SDK)
  │
  │  POST https://inference.mycompany.com/v1/generate
  │
  ▼
Cloud Load Balancer (external IP)
  │  Layer 4 TCP — just forwards packets to any healthy node
  ▼
Ingress Controller Pod (nginx)
  │  Layer 7 HTTP — reads Host header and path
  │  Matches rule: inference.mycompany.com /v1/generate → inference-svc
  ▼
Service: inference-svc (ClusterIP: 10.96.0.45)
  │  kube-proxy (iptables/IPVS rules on every node)
  │  Round-robin selects one of the 4 ready pod IPs
  ▼
Pod: inference-server-7d9f-xkp2 (Node: node-gpu-01, IP: 10.244.1.5:8000)
  │  Container receives HTTP request
  │  vLLM runs forward pass on GPU
  │  Streams tokens back
  ▼
Response flows back through Service → Ingress → Load Balancer → Client

Key points:
- Service selection is done by kube-proxy using iptables rules on the node itself
  (the packet never goes through a central proxy)
- Only pods that have passed their readiness probe are in the Service endpoint list
- If a pod crashes mid-request, the client gets a connection reset — app must retry
```

---

### What Happens When Traffic Spikes

```
1. RPS per pod crosses 20 (HPA custom metric threshold)
2. HPA controller adds +2 replicas to the Deployment
3. Scheduler finds nodes with free GPU slots
4. If no node has a free GPU:
     Cluster Autoscaler sees pending pods → provisions new GPU node from cloud
     (takes ~3-5 min for a new node to join)
5. New pods start, model loads (~60s, covered by readinessProbe delay)
6. Once ready, Service adds new pods to endpoint list
7. Traffic automatically distributes across old + new pods
```

Scale-down is the reverse: idle pods removed, empty nodes terminated by Cluster Autoscaler after a cooldown period (default 10 min).
