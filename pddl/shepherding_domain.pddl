(define (domain shepherding)
  (:requirements :strips :typing)

  (:types zone sheep)

  (:predicates
    ;; Locations
    (robot-at-zone ?z - zone)
    (flock-at-zone ?z - zone)
    (sheep-at-zone ?s - sheep ?z - zone)

    ;; Graph topology
    (zone-adjacent ?z1 - zone ?z2 - zone)
    (zone-is-goal ?z - zone)

    ;; State flags
    (flock-dispersed)
    (flock-compact)
    (flock-at-goal)

    ;; Spatial relationship for driving
    ;; True when zone ?rz is behind flock at ?fz
    ;; relative to the goal zone ?gz
    (behind-flock ?rz - zone ?fz - zone ?gz - zone)
  )

  ;; -------------------------------------------------------
  ;; Move robot to an adjacent zone
  ;; -------------------------------------------------------
  (:action move-robot
    :parameters (?from - zone ?to - zone)
    :precondition (and
      (robot-at-zone ?from)
      (zone-adjacent ?from ?to)
    )
    :effect (and
      (robot-at-zone ?to)
      (not (robot-at-zone ?from))
    )
  )

  ;; -------------------------------------------------------
  ;; Collect an outlier sheep back toward the flock
  ;; Robot must be in the same zone as the outlier sheep,
  ;; and the flock must be dispersed.
  ;; -------------------------------------------------------
  (:action collect-outlier
    :parameters (?s - sheep ?rz - zone ?fz - zone)
    :precondition (and
      (flock-dispersed)
      (robot-at-zone ?rz)
      (sheep-at-zone ?s ?rz)
      (flock-at-zone ?fz)
      (zone-adjacent ?rz ?fz)
    )
    :effect (and
      (sheep-at-zone ?s ?fz)
      (not (sheep-at-zone ?s ?rz))
      (flock-compact)
      (not (flock-dispersed))
    )
  )

  ;; -------------------------------------------------------
  ;; Drive the entire flock one zone toward the goal.
  ;; Robot must be behind the flock (opposite side from goal).
  ;; -------------------------------------------------------
  (:action drive-flock
    :parameters (?rz - zone ?fz - zone ?next-fz - zone ?gz - zone)
    :precondition (and
      (flock-compact)
      (robot-at-zone ?rz)
      (flock-at-zone ?fz)
      (zone-is-goal ?gz)
      (behind-flock ?rz ?fz ?gz)
      (zone-adjacent ?fz ?next-fz)
    )
    :effect (and
      (flock-at-zone ?next-fz)
      (not (flock-at-zone ?fz))
    )
  )

  ;; -------------------------------------------------------
  ;; Pen the flock when it reaches the goal zone
  ;; -------------------------------------------------------
  (:action pen-flock
    :parameters (?fz - zone)
    :precondition (and
      (flock-compact)
      (flock-at-zone ?fz)
      (zone-is-goal ?fz)
    )
    :effect (flock-at-goal)
  )
)
