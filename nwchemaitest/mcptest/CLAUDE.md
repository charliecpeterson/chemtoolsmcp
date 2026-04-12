  # NWChem Agent Workspace — Mac                                                                                                               
   
  ## How NWChem Runs Here                                                                                                                      
  - **Runner profile:** `docker_mac`                        
  - **Job directory:** `/Users/charlie/test/mcptest/`                                                                                          
  - **Runner profiles:** `/Users/charlie/test/mcptest/runner_profiles.json`
                                                                                                                                               
  ## Rules                                                  
  - Always use profile `docker_mac` when launching jobs                                                                                        
  - Use `chemtools` MCP tools for all parsing, input generation, and run control
  - After creating or editing an input: lint it with `chemtools_lint_nwchem_input`                                                             
  - For TCE jobs: inspect orbital ordering with `chemtools_parse_nwchem_movecs` before freezing  
