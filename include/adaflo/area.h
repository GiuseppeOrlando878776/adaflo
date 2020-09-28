#ifndef __area_h
#define __area_h

#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/distributed/tria.h>

#include <adaflo/parameters.h>

using namespace dealii;


template<unsigned int dim>
class Area {
public:
  /*! \brief Class constructor */
  Area(const FlowParameters &parameters,
       parallel::distributed::Triangulation<dim> &triangulation,
       std::shared_ptr<TimerOutput> external_timer)

private:
  void setup_dofs(); /*! \brief Set dofs */

  void mark_cells_for_refinement(); /*! \brief Mark cells for refinement */

  MPI_Comm communicator; /*! \brief Variable to store communicator (useful only for clearness) */

  FlowParameters parameters; /*! \brief Variable to store flow parameters */

  parallel::distributed::Triangulation<dim>& triangulation; /*! \brief Mesh storing */
  DoFHandler<dim>                            dof_handler_area;   /*! \brief DOF handler for area */
  const FESystem<dim>                        fe_area;  /*! \brief Finite eLement system for area */

  LinearAlgebra::distributed::BlockSparseMatrix<double> system_matrix; /*! \brief System matrix (each block is the cell-wise matrix) */
  LinearAlgebra::distributed::BlockVector<double> solution, solution_old; /*! \brief Current and previous solution */
  LinearAlgebra::distributed::BlockVector<double> rhs; /*! \brief Right-hand side */

  IndexSet locally_owned_dofs_area; /*! \brief Auxiliary variable to store owned DoFs */
  IndexSet locally_relevant_dofs_area; /*! \brief Auxiliary variable to store relevant DoFs */

  std::shared_ptr<TimerOutput> timer;         /*! \brief Timer output for printing */

  ConditionalOStream  pcout;       /*! \brief Output stream only for rank 0 */
};

// @sect{Area::Area}

// Class constructor. This basically initializes some variables that store already known infos
// like the triangulation or the flow parameters
template<unsigned int dim>
Area<dim>::Area(const FlowParameters &parameters,
                parallel::distributed::Triangulation<dim> &triangulation,
                std::shared_ptr<TimerOutput> external_timer):
                communicator(triangulation.get_communicator()),
                parameters(parameters), triangulation(triangulation),
                pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0),
                dof_handler_area(triangulation),
                fe_area(FE_DGQ<dim>(parameters.degree), 1),
                timer(new TimerOutput(pcout, TimerOutput::summary, TimerOutput::wall_times)) {}

// @sect{Area::setup_dofs}

// This function sets the degrees of freedom related to the area problem
template<unsigned int dim>
void Area<dim>::setup_dofs() {
  timer->enter_subsection("Setting DoFs for the area");

  /*--- Distribute DoFs ---*/
  dof_handler_area.distribute_dofs(fe_area);
  DoFRenumbering::component_wise(dof_handler_area);

  /*--- Set the number of locally owned and relevant DoFs ---*/
  locally_owned_dofs_area = dof_handler_area.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_area, locally_relevant_dofs_area);

  /*--- Initialize vectors (solution, old_solution and rhs) ---*/
  solution.reinit(locally_owned_dofs_area, communicator);
  solution.collect_sizes(); // The block is still empty so we have to call this function to make available the size
  old_solution.reinit(solution);
  old_solution.collect_sizes();
  rhs.reinit(locally_owned_dofs_area, communicator);
  rhs.collect_sizes();

  /*--- Initialize the system block matrix ---*/
  DynamicSparsityPattern dsp(locally_relevant_dofs_area);
  DoFTools::make_sparsity_pattern(dof_handler_area, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp, dof_handler_area.n_locally_owned_dofs_per_processor(),
                                             communicator, locally_relevant_dofs_area);
  system_matrix.reinit(locally_owned_dofs_area, dsp, communicator);

  timer->leave_subsection();
}
