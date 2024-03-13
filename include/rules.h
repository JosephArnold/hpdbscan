/*
* Copyright (c) 2018
* Markus Goetz
*
* This software may be modified and distributed under the terms of MIT-style license.
*
* Description: Cluster remapping rules
*
* Maintainer: m.goetz
*
* Email: markus.goetz@kit.edu
*/

#ifndef RULES_H
#define RULES_H

#include <omp.h>
#include <unordered_map>

#include "constants.h"

template<typename index_type>
class Rules {
    using RCluster = Cluster<index_type>;
    std::unordered_map<RCluster, RCluster> m_rules;

public:
    Rules() {
        m_rules[NOISE<index_type>] = 0;
    }

    inline const typename std::unordered_map<RCluster, RCluster>::const_iterator begin() const {
        return m_rules.begin();
    }

    inline const typename std::unordered_map<RCluster, RCluster>::const_iterator end() const {
        return m_rules.end();
    }

    inline void remove(const RCluster index) {
        m_rules.erase(m_rules.find(index));
    }

    RCluster rule(const RCluster cluster) const {
        const auto& pair = m_rules.find(cluster);
        if (pair != m_rules.end()) {
            return pair->second;
        }
        return NOT_VISITED<index_type>;
    }

    inline std::size_t size() const {
        return m_rules.size();
    }

    bool update(const RCluster first, const RCluster second) {
        if (first <= second or first >= NOISE<index_type>) {
            return false;
        }
        const auto& pair = m_rules.find(first);
        if (pair != m_rules.end()) {
            if (pair->second > second) {
                update(pair->second, second);
                m_rules[first] = second;
            } else {
                update(second, pair->second );
            }
        } else {
            m_rules[first]  = second;
        }

        return true;
    }
};

template<typename index_type>
void merge(Rules<index_type>& omp_out, Rules<index_type>& omp_in) {
    for (const auto& rule : omp_in) {
        omp_out.update(rule.first, rule.second);
    }
}

#pragma omp declare reduction(merge: Rules<std::int16_t>: merge(omp_out, omp_in)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(merge: Rules<std::int32_t>: merge(omp_out, omp_in)) initializer(omp_priv(omp_orig))
#pragma omp declare reduction(merge: Rules<std::int64_t>: merge(omp_out, omp_in)) initializer(omp_priv(omp_orig))

#endif // RULES_H
