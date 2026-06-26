# simnibs-invesalius-server

External processing server that runs [SimNIBS](https://simnibs.github.io/simnibs/) head modelling and TMS electric-field simulations, and relays results back to [InVesalius](https://github.com/invesalius/invesalius3) via Socket.IO.

---

## Structure

```
simnibs-invesalius-server/
├── relay_server.py
├── server.py
├── src/
│   └── simnibs_server/
│       ├── core/
│       │   ├── socket_client.py 
│       │   ├── message_handler.py
│       │   └── message_emit.py
│       └── processing/
│           ├── charm_runner.py
│           └── nifti_loader.py
├── requirements.txt
└── README.md
```

---

## Message protocol

All messages follow the InVesalius pubsub format: `{"topic": "...", "data": {...}}`.

---

## Requirements

- Python 3.10+
- SimNIBS 4.x installed with its Python environment active — `simnibs.segmentation.charm_main` and `simnibs.simulation` must be importable.
- InVesalius 3
